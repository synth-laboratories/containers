"""Job-scoped MCP access to the existing bounded annotation inspection tools."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading


@contextmanager
def serve_inspection_tools(tools):
    path = '/mcp/' + secrets.token_urlsafe(32)
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            if self.path != path:
                self.send_error(404)
                return
            size = int(self.headers.get('Content-Length', '0'))
            if size < 1 or size > 131072:
                self.send_error(413)
                return
            try:
                request = json.loads(self.rfile.read(size))
                method = request.get('method')
                ident = request.get('id')
                if ident is None:
                    self.send_response(202)
                    self.end_headers()
                    return
                if method == 'initialize':
                    result = {'protocolVersion': '2025-03-26', 'capabilities': {'tools': {}}, 'serverInfo': {'name':'synth-trace-inspection','version':'1'}}
                elif method == 'tools/list':
                    result = {'tools': [{'name':s['name'],'description':s['description'],'annotations':{'readOnlyHint':True,'destructiveHint':False,'idempotentHint':True,'openWorldHint':False},'inputSchema':s.get('inputSchema',s.get('input_schema',s.get('parameters',{})))} for s in tools.specs()]}
                elif method == 'tools/call':
                    params = request.get('params') or {}
                    try:
                        with lock:
                            output = tools.call(params['name'], params.get('arguments') or {})
                        result = {'content':[{'type':'text','text':json.dumps(output)}]}
                    except Exception as error:
                        result = {'isError':True,'content':[{'type':'text','text':str(error)}]}
                elif method == 'ping':
                    result = {}
                else:
                    self.send_json({'jsonrpc':'2.0','id':ident,'error':{'code':-32601,'message':'Unsupported method'}})
                    return
                self.send_json({'jsonrpc':'2.0','id':ident,'result':result})
            except (ValueError, TypeError, KeyError):
                self.send_error(400)
        def send_json(self, result):
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self): self.send_error(405)
        def do_DELETE(self):
            self.send_response(200 if self.path == path else 404)
            self.end_headers()
    server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}{path}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
