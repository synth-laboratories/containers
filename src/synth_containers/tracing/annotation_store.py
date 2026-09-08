"""Local append-only annotation revisions using the canonical V5 evidence API."""
from __future__ import annotations

import fcntl
import json
import os
import uuid
from pathlib import Path

from .canonical import utc_now, content_digest
from .evidence_ops import attach_many
from .models.selectors import TraceSelectorV1, resolve_selector
from .models.standards import AnnotationV1, ProducerRefV1, TraceAnnotatorDefinitionV1, AnnotationInspectionV1
from .validation.rehydrate import evidence_bundle_from_payload, build
from .validation.validator import validate_evidence
from .projections.visual import visual_from_sealed


class AnnotationStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, document):
        # A revised trace gets its own head; old notes never silently migrate.
        path = self.root / document.content_digest.replace(':', '-')
        path.mkdir(exist_ok=True)
        return path

    def load(self, document, base):
        path = self._directory(document) / 'head.json'
        if not path.exists():
            return base
        head = json.loads(path.read_text())
        name = str(head['file'])
        if Path(name).name != name:
            raise ValueError('Invalid evidence head')
        bundle = evidence_bundle_from_payload(json.loads((path.parent / name).read_text()))
        if content_digest(bundle) != bundle.content_digest or bundle.trace_ref.content_digest != document.content_digest:
            raise ValueError('Evidence integrity mismatch')
        return bundle

    def append(self, document, base, request):
        directory = self._directory(document)
        with (directory / 'write.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            bundle = self.load(document, base)
            if request.get('expected_digest') != bundle.content_digest:
                raise ValueError('Evidence changed; refresh before saving another revision')
            target = build(TraceSelectorV1, request['target'])
            resolution = resolve_selector(document, target)
            if not resolution.resolved:
                raise ValueError(f'Unresolved annotation target: {resolution.reason}')
            body = str(request.get('body', '')).strip()
            if not body or len(body) > 12000:
                raise ValueError('Annotation body must contain 1–12000 characters')
            kind = request.get('author_kind', 'human')
            name = str(request.get('author', '')).strip()
            if kind not in ('human', 'model') or not name or len(name) > 120:
                raise ValueError('A human/model author and name are required')
            review = request.get('review_state', 'unreviewed')
            if review not in ('unreviewed', 'accepted', 'rejected', 'needs_review', 'disputed'):
                raise ValueError('Invalid review state')
            previous_id = request.get('supersedes_id')
            previous = next((note for note in bundle.annotations if note.annotation_id == previous_id), None)
            if previous_id:
                if previous is None or previous.target != target:
                    raise ValueError('Supersession must retain an existing annotation target')
                if any(note.supersedes_id == previous_id for note in bundle.annotations):
                    raise ValueError('Annotation already superseded; refresh first')
            label = request.get('label', 'note')
            taxonomy = ('note', 'failure', 'coordination', 'reward', 'capture-gap')
            if label not in taxonomy:
                raise ValueError('Unknown annotation label')
            definition = TraceAnnotatorDefinitionV1(
                annotator_id=f'workshop-review-{target.kind}-v1', name='Workshop evidence review',
                purpose='Describe retained source evidence without modifying the rollout',
                taxonomy=taxonomy, required_subject_scope=str(target.kind), grounding_requirement='summary_allowed',
            ).sealed()
            projection = visual_from_sealed(document, bundle).to_dict()
            projection_digest = content_digest(projection)
            inspection_manifest = {'schema_version': 'synth.annotation-inspection.v1', 'trace_digest': document.content_digest, 'projection_digest': projection_digest, 'projection_id': 'synth.trace-visual.v1'}
            inspection_digest = content_digest(inspection_manifest)
            annotation = AnnotationV1(
                annotation_id=f'annotation_{uuid.uuid4().hex}', annotator_id=definition.annotator_id,
                annotator_version=definition.version, annotator_digest=definition.content_digest,
                target=target, annotation_type='evidence_note', labels=(label,),
                author_kind=kind, producer=ProducerRefV1(kind=kind, name=name),
                created_at=utc_now(), grounding='summary_only', rationale=body,
                inspected_projection='synth.trace-visual.v1', inspection=AnnotationInspectionV1(source='projection', trace_body_read=False, projection_id='synth.trace-visual.v1', projection_digest=projection_digest, projection_manifest_digest=inspection_digest),
                evidence=(target,), status='applied', review_state=review,
                revision=previous.revision + 1 if previous else 1, supersedes_id=previous_id,
            ).sealed()
            records = []
            if not any(d.annotator_id == definition.annotator_id for d in bundle.annotator_definitions):
                records.append(('annotator_definition', definition))
            records.append(('annotation', annotation))
            updated = attach_many(bundle, records=tuple(records))
            errors = [f.to_dict() for f in validate_evidence(document, updated)[0] if str(f.severity) == 'error']
            if errors:
                raise ValueError(errors)
            for digest, value in ((projection_digest, projection), (inspection_digest, inspection_manifest)):
                (directory / ('inspection-' + digest.replace(':', '-') + '.json')).write_text(json.dumps(value, separators=(',', ':')))
            # Retain both sides of the append, atomically advance only the head.
            for record in (bundle, updated):
                name = record.content_digest.replace(':', '-') + '.json'
                path = directory / name
                if not path.exists():
                    path.write_text(json.dumps(record.to_dict(), separators=(',', ':')))
            temporary = directory / f'head-{uuid.uuid4().hex}.tmp'
            temporary.write_text(json.dumps({'file': updated.content_digest.replace(':', '-') + '.json'}))
            os.replace(temporary, directory / 'head.json')
            return updated, annotation
