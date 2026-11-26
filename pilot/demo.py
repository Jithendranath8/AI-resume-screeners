"""Local presentation demo. Run: .venv/bin/python pilot/demo.py"""
import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from experiment import KEYWORDS, dataset, normalize

ROOT = Path(__file__).resolve().parent


class Detector:
    def __init__(self):
        self.run = json.loads((ROOT / 'results/run.json').read_text())
        for name, expected in self.run['input_sha256'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Recorded experiment input changed: {name}')
        train = [r for r in dataset() if r['split'] == 'train']
        self.model = make_pipeline(
            TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5)),
            LogisticRegression(C=1, class_weight='balanced', solver='liblinear',
                               random_state=42, max_iter=1000))
        self.model.fit([normalize(r['text']) for r in train], [r['label'] for r in train])
        self.threshold = self.run['thresholds']['normalized_char_lr']

    def analyze(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Enter at least one non-empty line of text.')
        if len(text) > 16000:
            raise ValueError('Use at most 16,000 characters.')
        lines = [(i, line.strip()) for i, line in enumerate(text.splitlines(), 1) if line.strip()]
        if len(lines) > 100 or any(len(line) > 2000 for _, line in lines):
            raise ValueError('Use at most 100 lines and 2,000 characters per line.')
        scores = self.model.predict_proba([normalize(line) for _, line in lines])[:, 1]
        result = []
        for (number, line), score in zip(lines, scores):
            matches = [k for k in KEYWORDS if k in line.lower()]
            result.append(dict(line=number, text=line, normalized=normalize(line),
                               score=float(score), flagged=bool(score >= self.threshold),
                               keyword_flagged=bool(matches), keyword_matches=matches,
                               normalization_changed=normalize(line) != line))
        return dict(lines=result, threshold=self.threshold,
                    flagged_lines=sum(r['flagged'] for r in result),
                    keyword_flagged_lines=sum(r['keyword_flagged'] for r in result),
                    recorded_metrics=self.run['metrics']['normalized_char_lr'])


def handler(detector):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, content, kind='application/json; charset=utf-8'):
            data = content.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/':
                self.respond(200, (ROOT / 'demo.html').read_text(), 'text/html; charset=utf-8')
            elif self.path == '/health':
                self.respond(200, json.dumps({'ready': True}))
            else:
                self.respond(404, json.dumps({'error': 'Not found'}))

        def do_POST(self):
            if self.path != '/analyze':
                self.respond(404, json.dumps({'error': 'Not found'}))
                return
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 100000:
                    raise ValueError('Request is empty or too large.')
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError('Expected a JSON object.')
                result = detector.analyze(payload.get('text'))
                self.respond(200, json.dumps(result))
            except (ValueError, UnicodeError) as exc:
                self.respond(400, json.dumps({'error': str(exc)}))

        def log_message(self, *_):
            pass  # Do not persist pasted text or request logs.
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(Detector()))
    print(f'Demo ready: http://127.0.0.1:{args.port} — Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
