import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { defineConfig } from 'vite'

// The built page is served by scanner/scanner_app.py. `npm run dev` proxies the scanner's endpoints to it.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(import.meta.dirname, 'src') } },
  server: { proxy: { '/health': 'http://127.0.0.1:8502', '/scan': { target: 'http://127.0.0.1:8502', bypass: (req) => (req.method === 'GET' ? '/index.html' : undefined) } } },
})
