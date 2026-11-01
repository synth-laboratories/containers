"""Visible, revision-fenced controls for durable steering and follow-up input."""

import asyncio
import json
import os
import time

from prompt_toolkit.mouse_events import MouseEventType

from .picker import Picker
from .rendering import safe

CHILD_LABEL = {"child_result": "Child result", "child_message": "Child message", "agent_message": "Agent message"}


def is_child(item):
    return item.get("view") == "subagents" or (item.get("view") is None and (item.get("origin") or {}).get("kind") in {"child_result", "child_message", "agent_message"})


def child_rows(children, width, action=None):
    """Keep every notification addressable; steering never edits its payload."""
    rows = []
    for kind, index, item in children:
        origin = item["origin"]
        label = CHILD_LABEL.get(origin.get("kind"), "Child")
        bits = [f'{label} · {safe(str(origin.get("name") or ""))}']
        if origin.get("kind") == "child_result" and origin.get("status"):
            bits[0] += f" {safe(str(origin['status']))}"
        bits.append("steer" if kind == "steering" else "queued")
        tasks = origin.get("tasks") or []
        if tasks:
            bits.append("task " + ", ".join(safe(str(t).rpartition(":")[2] if str(t).startswith(str(origin.get("child") or "\0")) else str(t)) for t in tasks))
        preview = " ".join(safe(str(origin.get("preview") or "")).split())
        line = " " + " · ".join(bits)
        room = width - len(line) - (13 if action else 4)
        if preview and room > 8:
            line += " · " + (preview if len(preview) <= room else preview[:room - 1] + "…")
        if action:
            line = line[:max(0, width - 8)]
        rows.append(("class:watch-dim", line))
        if action:
            rows.append(("class:picker-title", " Steer ", action(kind, index, item)))
        rows.append(("class:watch-dim", "\n"))
    return rows


class QueueMixin:
    def init_queue(self):
        self.pending_followup = None
        self.queue = {"revision": None, "steering": [], "followUp": []}
        self.queue_mode = "steering"
        self.queue_filter = "all"
        self.queue_sender = None
        self.queue_edit = None
        self.queue_edit_pending = False
        self.queue_choice = None
        self.queue_generation = 0
        self.queue_error = None
        self.queue_draft = None

    def set_queue_view(self, view, sender=None):
        self.queue_filter, self.queue_sender = view, sender
        path = getattr(self, "queue_preferences", None)
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps({"view": view, "sender": sender}))
                temporary.replace(path)
            except OSError as error:
                self.notice(f"Queue preferences: {error}")

    def load_queue_view(self, path):
        self.queue_preferences = path
        try:
            value = json.loads(path.read_text())
            if value.get("view") in {"all", "human", "subagents", "peers"}:
                self.queue_filter = value["view"]
                self.queue_sender = value.get("sender") if isinstance(value.get("sender"), str) else None
        except (OSError, ValueError, TypeError):
            pass

    def stash_queue_draft(self):
        if self.queue_draft is None:
            self.queue_draft = (self.draft.document, list(self.attachments))
            self.attachments = []
        self.draft.reset()

    def restore_queue_draft(self):
        if self.queue_draft is not None:
            document, self.attachments = self.queue_draft
            self.queue_draft = None
            self.draft.document = document

    def double_enter_steer(self, draft):
        pending = self.pending_followup
        if (pending and not draft.text.strip() and time.monotonic() <= pending["until"]
                and (self.thread, self.turn) == (pending["thread"], pending["turn"])):
            pending["steer"] = True
            pending["ready"].set()
            return True
        return False

    def defer_followup(self, text):
        # Wait before the one and only host mutation: promoting after enqueue
        # would require a separate remove/send and could deliver twice.
        pending = {"text": text, "thread": self.thread, "turn": self.turn, "steer": False,
                   "ready": asyncio.Event(), "until": time.monotonic() + 0.4}
        previous = self.pending_followup
        self.pending_followup = pending
        if previous:
            previous["ready"].set()
        self.start(self.deliver_followup(pending))

    async def deliver_followup(self, pending):
        try:
            await asyncio.wait_for(pending["ready"].wait(), max(0, pending["until"] - time.monotonic()))
        except TimeoutError:
            pass
        if self.pending_followup is pending:
            self.pending_followup = None
        if self.thread != pending["thread"] or self.turn not in (None, pending["turn"]):
            self.notice("Message not sent: session or turn changed\n" + pending["text"])
            return
        prefix = ("/steer " if pending["steer"] else "/followup ") if self.turn else ""
        await self.input(prefix + pending["text"])

    async def refresh_queue(self):
        if not self.native_settings:
            return
        self.queue_generation += 1
        generation, thread = self.queue_generation, self.thread
        try:
            value = await self.backend.request("mlok/session/queue/view", threadId=thread)
            # A single revision fences the projection; a concurrent mutation
            # retries on the next refresh rather than mixing pages.
            while value.get("nextOffset") is not None:
                page = await self.backend.request("mlok/session/queue/view", threadId=thread,
                    offset=value["nextOffset"], expectedRevision=value["revision"])
                value["entries"].extend(page["entries"])
                value["nextOffset"] = page.get("nextOffset")
        except (ValueError, RuntimeError, OSError, ConnectionError, TimeoutError) as error:
            if generation == self.queue_generation and thread == self.thread:
                self.queue_error = "Queue unavailable" if "not supported" in str(error) else "Queue disconnected"
                self.app.invalidate()
            return
        if generation == self.queue_generation and thread == self.thread:
            self.queue = value
            self.queue_error = None
            self.app.invalidate()

    def queue_all(self):
        return [("steering", i, item) for i, item in enumerate(self.queue.get("steering", []))] + [
            ("follow_up", i, item) for i, item in enumerate(self.queue.get("followUp", []))
        ]

    def queue_entries(self):
        """User Steer/Follow-up items; child notifications are listed separately."""
        return [entry for entry in self.queue_all() if not is_child(entry[2])]

    def queue_children(self):
        return [entry for entry in self.queue_all() if is_child(entry[2])]

    def queue_visible(self):
        return self.native_settings and bool((self.turn and (self.queue.get("revision") or self.queue_error)) or self.queue_all() or self.queue_edit or self.message_entries())

    def message_entries(self):
        entries = [m for m in self.queue.get("entries", []) if m.get("state") in {"pending", "held"}]
        view = getattr(self, "queue_filter", "all")
        sender = getattr(self, "queue_sender", None)
        def matches(m):
            if not sender:
                return True
            if sender == "direct":
                return not m.get("channel")
            if sender.startswith("#"):
                return m.get("channel") == sender[1:]
            return sender in {m.get("sender"), (m.get("origin") or {}).get("name"), (m.get("origin") or {}).get("path"), m.get("parentSession")}
        return [m for m in entries if (view == "all" or m.get("view") == view) and matches(m)]

    def message_fragments(self):
        from prompt_toolkit.application import get_app

        rows = [("class:footer", " Queue ")]
        pending = [m for m in self.queue.get("entries", []) if m.get("state") in {"pending", "held"}]
        for view, label in [("all", "All"), ("human", "Human"), ("subagents", "Subagents"), ("peers", "Peers")]:
            count = sum(view == "all" or m.get("view") == view for m in pending)
            def select(event, view=view):
                if event.event_type == MouseEventType.MOUSE_UP:
                    self.set_queue_view(view)
                    self.app.invalidate()
            style = "class:picker-selected" if getattr(self, "queue_filter", "all") == view else "class:picker"
            rows.append((style, f" {label} {count} ", select))
        if getattr(self, "queue_filter", "all") == "peers":
            for name in ["direct", "#mlok"]:
                def filter_channel(event, name=name):
                    if event.event_type == MouseEventType.MOUSE_UP:
                        self.set_queue_view(self.queue_filter, name if self.queue_sender != name else None)
                        self.app.invalidate()
                rows.append(("class:picker-selected" if self.queue_sender == name else "class:picker", f" {name} ", filter_channel))
            muted = "mlok" in self.queue.get("mutedChannels", [])
            def toggle_mute(event):
                if event.event_type == MouseEventType.MOUSE_UP:
                    self.start(self.mute_channel(not muted))
            rows.append(("class:picker-title", " Unmute " if muted else " Mute ", toggle_mute))
        transport = self.queue.get("transport", {}).get("state")
        if transport:
            rows.append(("class:watch-dim", f" · MQ {safe(transport)}"))
        rows.append(("class:footer", "\n"))
        if self.turn:
            for label, mode in [("Steer", "steering"), ("Follow-up", "follow_up")]:
                def select_mode(event, mode=mode):
                    if event.event_type == MouseEventType.MOUSE_UP:
                        self.queue_mode = mode
                        self.app.invalidate()
                rows.append(("class:picker-selected" if self.queue_mode == mode else "class:picker", f" {label} ", select_mode))
            rows.append(("class:footer", "\n"))
        width = self.transcript_columns(get_app().output.get_size().columns)
        for index, item in enumerate(self.message_entries()[:2]):
            label = {"human": "Human", "subagents": "Subagent", "peers": "Peer"}.get(item.get("view"), "Message")
            origin = item.get("origin") or {}
            sender = safe(str(item.get("senderName") or origin.get("name") or item.get("sender") or ""))
            target = "you" if item.get("audience") == "human" else item.get("channel") or "agent"
            text = " ".join(safe(item.get("text") or "").split())
            line = f" {label} · {sender} → {target} · {text}"
            if item.get("view") == "human" and item.get("audience") == "agent":
                line = f" {'Steer' if item.get('delivery') == 'steering' else 'Follow-up'} {index + 1} · {text}"
            delivery = "Steer" if item.get("delivery") == "steering" else "Follow-up"
            suffix = f" · {delivery} · {item['state']}"
            if item.get("publicationError"):
                suffix += " · send failed"
            line = line[:max(4, width - len(suffix) - 1)] + suffix
            def inspect(event, item=item):
                if event.event_type == MouseEventType.MOUSE_UP:
                    self.open_message(item)
                    self.app.invalidate()
            rows.append(("class:watch-dim", line + "\n", inspect))
        if len(self.message_entries()) > 2:
            def expand(event):
                if event.event_type == MouseEventType.MOUSE_UP:
                    self.open_queue_picker()
                    self.app.invalidate()
            rows.append(("class:picker-title", f" +{len(self.message_entries()) - 2} messages ", expand))
        return rows

    async def mute_channel(self, muted):
        await self.backend.request("mlok/messages/mute", threadId=self.thread, channel="mlok", muted=muted)
        await self.refresh_queue()

    def open_message(self, item):
        if self.picker is None:
            self.stash_queue_draft()
        self.queue_choice = {"item": item, "revision": self.queue.get("revision")}
        actions = [{"id": "inspect", "label": "Read", "detail": item.get("text", "")}]
        if item.get("view") != "human":
            actions.append({"id": "reply", "label": "Reply"})
        if item.get("state") == "pending":
            actions.append({"id": "hold", "label": "Hold"})
        elif item.get("state") == "held":
            actions.append({"id": "release", "label": "Release"})
        actions.append({"id": "dismiss", "label": "Dismiss"})
        if item.get("view") == "human" and item.get("state") == "pending" and item.get("audience") == "agent":
            actions += [{"id": "edit", "label": "Edit"}, {"id": "steer", "label": "Steer now"}, {"id": "up", "label": "Move up"}, {"id": "down", "label": "Move down"}]
        if item.get("view") == "human" and item.get("state") == "pending" and item.get("audience") == "agent":
            actions = [{"id": "edit", "label": "Edit"}, {"id": "up", "label": "Move up"}, {"id": "down", "label": "Move down"}, {"id": "remove", "label": "Remove"}, {"id": "steer", "label": "Steer now"}] + [a for a in actions if a["id"] in {"inspect", "hold"}]
        self.picker = Picker("Queue actions", actions)
        self.draft.reset()

    def queue_fragments(self):
        from prompt_toolkit.application import get_app

        def mode(value):
            def clicked(event):
                if event.event_type == MouseEventType.MOUSE_UP:
                    self.queue_mode = value
                    self.app.invalidate()
            return clicked

        def action(kind, index, operation, revision, item):
            def clicked(event):
                if event.event_type == MouseEventType.MOUSE_UP:
                    self.start(self.queue_action(kind, index, operation, revision, item))
            return clicked

        if self.queue.get("schemaVersion") == 2 and not self.queue_error:
            return ([("class:picker-title", " Editing queued message\n")] if self.queue_edit else []) + self.message_fragments()
        rows = []
        if self.queue_error:
            return [("class:footer-manual", " " + self.queue_error)]
        if self.turn:
            for label, value in [("Steer", "steering"), ("Follow-up", "follow_up")]:
                rows.append(("class:picker-selected" if self.queue_mode == value else "class:picker", " " + label + " ", mode(value)))
            rows.append(("class:footer", "\n"))
        if self.queue_edit:
            rows.append(("class:picker-title", " Editing queued message\n"))
        entries = self.queue_entries()
        if self.queue_all() and not self.turn:
            def drain(event):
                if event.event_type == MouseEventType.MOUSE_UP:
                    self.start(self.run_queue())
            rows.append(("class:picker-title", " Run queued ", drain))
            rows.append(("class:footer", "\n"))
        width = self.transcript_columns(get_app().output.get_size().columns)
        for kind, index, item in entries[:3]:
            name = "Steer" if kind == "steering" else "Follow-up"
            value = " ".join(safe(item.get("text") or "").split())
            if item.get("imageCount"):
                value += f" · {item['imageCount']} image{'s' if item['imageCount'] != 1 else ''}"
            compact = width < 65
            limit = max(4, width - (38 if compact else 54) - (8 if kind == "follow_up" else 0))
            if len(value) > limit:
                value = value[:limit - 1] + "…"
            revision = self.queue.get("revision")
            rows.append(("class:footer", f" {name} {index + 1} · {value} "))
            for label, operation in [(("E" if compact else "Edit"), "edit"), ("↑", "up"), ("↓", "down"), ("×", "remove")]:
                rows.append(("class:picker-title", f" {label} ", action(kind, index, operation, revision, item)))
            rows.append(("class:picker-title", " Steer ", action(kind, index, "steer", revision, item)))
            rows.append(("class:footer", "\n"))
        children = self.queue_children()[:max(0, 5 - min(3, len(entries)))]
        rows.extend(child_rows(children, width,
                               lambda kind, index, item: action(kind, index, "steer", self.queue.get("revision"), item)))
        hidden = len(self.queue_all()) - min(3, len(entries)) - len(children)
        if hidden:
            def manage(event):
                if event.event_type == MouseEventType.MOUSE_UP:
                    try:
                        self.open_queue_picker()
                    except ValueError as error:
                        self.notice(error)
                    self.app.invalidate()
            rows.append(("class:picker-title", f" +{hidden} queued ", manage))
        return rows

    def open_queue_picker(self):
        if self.queue_error:
            raise ValueError(self.queue_error)
        if self.queue.get("schemaVersion") == 2:
            entries = self.message_entries()
            if not entries:
                raise ValueError("Queue is empty")
            self.stash_queue_draft()
            self.picker = Picker("Queue", [{"id": m["id"], "label": (' '.join(safe(m.get('text') or '').split())[:100] if m.get("view") == "human" else f"{m.get('view')} · {m.get('sender')} · {' '.join(safe(m.get('text') or '').split())[:100]}"), "detail": ("Steer" if m.get("delivery") == "steering" else "Follow-up") + (" · held" if m["state"] == "held" else ""), "item": m} for m in entries])
            self.draft.reset()
            return
        entries = self.queue_all()
        if not entries:
            raise ValueError("Queue is empty")
        revision = self.queue.get("revision")
        options = []
        if not self.turn:
            options.append({"id": "run", "label": "Run queued", "detail": "Pending messages"})
        for kind, index, item in entries:
            origin = item.get("origin") or {}
            label = " ".join(safe(item.get("text") or origin.get("preview") or "").split())[:100] or "Image"
            if is_child(item):
                label = f"{CHILD_LABEL.get(origin.get('kind'), 'Child')} · {safe(str(origin.get('name') or ''))} · {label}"
            options.append({"id": f"{kind}:{index}", "label": label,
                            "detail": "Steer" if kind == "steering" else "Follow-up",
                            "kind": kind, "index": index, "revision": revision, "item": item})
        self.picker = Picker("Queue", options)
        self.draft.reset()

    async def choose_queue(self):
        option = self.picker.current(self.draft.text)
        if not option:
            return
        if self.picker.kind == "Message" or (self.picker.kind == "Queue actions" and self.queue.get("schemaVersion") == 2):
            choice = self.queue_choice
            item, action = choice["item"], option["id"]
            self.picker = None
            self.draft.reset()
            self.restore_queue_draft()
            if action == "inspect":
                try:
                    await self.backend.request("mlok/messages/action", threadId=self.thread, messageId=item["id"], expectedRevision=choice["revision"], action="read")
                except ValueError:
                    pass  # a newer queue revision cannot prevent local inspection
                await self.refresh_queue()
                self.notice(item.get("text") or "", error=False)
            elif action == "reply":
                if self.draft.text or self.attachments:
                    raise ValueError("Current draft retained; reply not started")
                target = (item.get("origin") or {}).get("sender") or item.get("sender")
                self.message_reply_to = item["id"]
                self.draft.text = f"/message {item.get('replyTarget') or ('mq:' + target)} "
                self.draft.cursor_position = len(self.draft.text)
            elif action in {"edit", "steer", "up", "down", "remove"}:
                await self.queue_action(item["delivery"], 0, action, choice["revision"], item)
            else:
                await self.backend.request("mlok/messages/action", threadId=self.thread, messageId=item["id"], expectedRevision=choice["revision"], action=action)
                await self.refresh_queue()
            self.app.invalidate()
            return
        if self.picker.kind == "Queue":
            if self.queue.get("schemaVersion") == 2:
                self.open_message(option["item"])
                self.app.invalidate()
                return
            if option["id"] == "run":
                self.picker = None
                self.draft.reset()
                await self.run_queue()
                return
            self.queue_choice = option
            self.picker = Picker("Queue actions", ([] if is_child(option["item"]) else [
                {"id": "edit", "label": "Edit"}, {"id": "up", "label": "Move up"},
                {"id": "down", "label": "Move down"}, {"id": "remove", "label": "Remove"},
            ]) + [{"id": "steer", "label": "Steer now"}])
            self.draft.reset()
        else:
            choice = self.queue_choice
            self.picker = None
            self.draft.reset()
            await self.queue_action(choice["kind"], choice["index"], option["id"], choice["revision"], choice["item"])
        self.app.invalidate()

    async def run_queue(self):
        if self.turn:
            raise ValueError("A turn is already running")
        result = await self.backend.request("mlok/session/drain", threadId=self.thread)
        self.adopt_turn_start(result["turnId"])
        self.app.invalidate()

    async def queue_action(self, kind, index, operation, revision, item):
        if operation == "edit":
            if self.draft.text or self.attachments:
                raise ValueError("Finish the current draft before editing the queue")
            self.queue_edit = {"kind": kind, "index": index, "revision": revision, **({"messageId": item["id"]} if item.get("id") else {})}
            self.draft.text = item.get("text") or ""
            self.draft.cursor_position = len(self.draft.text)
            self.app.layout.focus(self.draft)
            self.app.invalidate()
            return
        thread = self.thread
        try:
            await self.backend.request("mlok/session/queue/update", threadId=thread,
                expectedRevision=revision, kind=kind, index=index, action=operation, **({"messageId": item["id"]} if item.get("id") else {}))
        except ValueError as error:
            if operation == "steer" and any(value in str(error) for value in ("invalid queue action", "child notifications", "only a follow-up")):
                self.notice("Steer unavailable · host update required")
                return
            raise
        if self.thread != thread:
            return
        await self.refresh_queue()
        if operation == "steer" and not self.turn:
            await self.run_queue()

    async def commit_queue_edit(self, value, edit):
        try:
            await self.backend.request("mlok/session/queue/update", threadId=self.thread,
                expectedRevision=edit["revision"], kind=edit["kind"], index=edit["index"], action="edit", text=value, **({"messageId": edit["messageId"]} if edit.get("messageId") else {}))
        except (ValueError, RuntimeError, OSError, TimeoutError) as error:
            self.notice(error)
            if not self.draft.text:
                self.draft.text = value
            await self.refresh_queue()
            self.app.invalidate()
            return
        finally:
            self.queue_edit_pending = False
        self.queue_edit = None
        await self.refresh_queue()
