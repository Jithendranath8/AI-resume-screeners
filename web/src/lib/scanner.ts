// Types for what scanner/scanner_app.py returns, and the few helpers both pages share.

export type Level = 'attack' | 'review' | 'clean'

export interface Finding {
  title: string
  severity: 'attack' | 'review'
  technique_label: string | null
  intents: string[]
  text: string
  explanation: string
  channel: string
  page: number | null
  bbox: [number, number, number, number] | null
  hidden: boolean
  instruction: string | boolean | null
  agreement?: number | null
  ocr_saw?: string
  scores?: Record<string, number>
  obfuscated?: boolean
}

export interface PageSize { width: number; height: number }
export interface PageImage extends PageSize { number: number; image: string }

export interface Decision { valid: boolean; decision: 'SELECT' | 'REJECT'; score: number; reason: string; error?: string | null }

export interface Impact {
  available: boolean
  reason?: string
  model?: string
  provider?: string
  results?: { original?: Decision; as_read: Decision; cleaned: Decision }
  removed_blocks?: number
  skipped?: string[]
}

export interface Block { page: number | null; channel: string; text: string; agreement?: number | null; scores: Record<string, number>; finding?: unknown }

export interface Report {
  verdict: { level: Level; summary: string; attack_types: string[]; review_types: string[] }
  pages: number
  seconds: number
  findings: Finding[]
  checks: { name: string; result: number; detail: string }[]
  blocks: Block[]
  pages_rendered: PageImage[]
  test_attack: { label: string } | null
  impact?: Impact
}

export interface Health {
  version: string
  detectors: { name: string; threshold: number }[]
  max_pages: number
  attacks: { key: string; label: string }[]
  presets: Record<string, string>
  samples: { id: string; title: string; group: string }[]
  screener: { available: boolean; options: { id: string; model: string; provider: string; hosted: boolean }[] }
  limits: string[]
}

export const LEVELS: Record<Level, string> = { attack: 'Attack found', review: 'Needs review', clean: 'No attack found' }

export const isInstruction = (f: Finding) => Boolean(f.instruction) && f.severity === 'attack'

/** Where a finding's mark sits on its page, in percent. Text outside the page is pulled to the nearest edge. */
export function markBox(f: Finding, page: PageSize) {
  let [x0, y0, x1, y1] = f.bbox as [number, number, number, number]
  const outside = x0 >= page.width || x1 <= 0 || y0 >= page.height || y1 <= 0
  if (outside) {
    const w = Math.min(page.width * 0.42, Math.max(120, x1 - x0))
    x0 = x0 >= page.width ? page.width - w - 4 : Math.max(4, Math.min(x0, page.width - w - 4))
    x1 = x0 + w
    y0 = Math.max(4, Math.min(y0, page.height - 24))
    y1 = Math.max(y0 + 14, Math.min(y1, page.height - 4))
  } else if (f.hidden && x1 - x0 < page.width * 0.3) {
    const w = page.width * 0.3
    x0 = Math.max(0, Math.min(x0, page.width - w))
    x1 = x0 + w
  }
  return {
    outside,
    style: {
      left: `${(100 * x0) / page.width}%`,
      top: `${(100 * y0) / page.height}%`,
      width: `${(100 * (x1 - x0)) / page.width}%`,
      minHeight: `${Math.max(1.4, (100 * (y1 - y0)) / page.height)}%`,
    },
  }
}

export async function getHealth(): Promise<Health> {
  const response = await fetch('/health')
  if (!response.ok) throw new Error('The scanner did not answer.')
  return response.json()
}

export interface ScanOptions { attack: string; payload: string; images: boolean; impact: boolean; screener: string }

export async function scanFile(name: string, source: { file?: File; sample?: string }, o: ScanOptions): Promise<Report> {
  const params = new URLSearchParams({ name, attack: o.attack, payload: o.payload, images: o.images ? '1' : '0', impact: o.impact ? '1' : '0' })
  if (source.sample) params.set('sample', source.sample)
  if (o.screener) params.set('screener', o.screener)
  const response = await fetch('/scan?' + params, { method: 'POST', headers: { 'Content-Type': 'application/pdf' }, body: source.sample ? null : source.file })
  const data = await response.json()
  if (!response.ok) throw new Error(data.error || 'The scan failed.')
  return data
}

export function download(name: string, text: string, type: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([text], { type }))
  a.download = name
  document.body.append(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}
