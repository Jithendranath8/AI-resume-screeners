import { useCallback, useEffect, useId, useRef, useState } from 'react'

import { Dropzone } from '@/components/site/Dropzone'
import { Nav } from '@/components/site/Nav'
import { Segmented } from '@/components/site/Segmented'
import { Sheet } from '@/components/site/Sheet'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Textarea } from '@/components/ui/textarea'
import { type Decision, type Finding, type Health, type Level, type Report, LEVELS, download, getHealth, isInstruction, scanFile } from '@/lib/scanner'
import { cn } from '@/lib/utils'

interface Item {
  id: number
  name: string
  file?: File
  sample?: string
  status: 'waiting' | 'scanning' | 'done' | 'error'
  report?: Report
  error?: string
}

const OWN = 'own'
const AUTO = 'auto'
const STATUS: Record<Item['status'], string> = { waiting: 'Waiting', scanning: 'Scanning…', done: '', error: 'Could not be scanned' }
const LEVEL_TEXT: Record<Level, string> = { attack: 'text-attack', review: 'text-review', clean: 'text-clean' }
const LEVEL_RULE: Record<Level, string> = { attack: 'border-t-attack', review: 'border-t-review', clean: 'border-t-clean' }
const field = 'h-10 w-full bg-paper text-[15px]'
const panel = 'border-t border-line px-6 py-7 sm:px-8'
const h2 = 'mb-4 text-lg font-semibold leading-tight tracking-[-0.005em]'

/* ------------------------------------------------------------------ options */

interface Options { images: boolean; attack: string; preset: string; payload: string; impact: boolean; screener: string }

function OptionsPanel({ health, o, set }: { health: Health; o: Options; set: (patch: Partial<Options>) => void }) {
  const id = useId()
  const s = health.screener
  const chosen = s.options.find((x) => x.id === o.screener)
  const hosts = [...new Set(s.options.filter((x) => x.hosted).map((x) => x.provider))]
  const sentTo = chosen
    ? (chosen.hosted ? `When switched on, the extracted resume text is sent to ${chosen.provider}.` : 'Runs on this computer.')
    : hosts.length
      ? `When switched on, the extracted resume text is sent to ${hosts.length > 1 ? 'the first of ' + hosts.join(' or ') + ' that answers' : hosts[0]}.`
      : 'Runs on this computer.'
  return (
    <Accordion type="single" collapsible className="noprint mt-6 border-y border-line">
      <AccordionItem value="options">
        <AccordionTrigger className="py-3.5 text-[15px] font-medium hover:no-underline">Options and test attacks</AccordionTrigger>
        <AccordionContent className="grid gap-x-12 gap-y-7 pb-6 pt-1 text-[15px] md:grid-cols-3">
          <fieldset className="space-y-3">
            <legend className="mb-2 font-semibold">Scanning</legend>
            <div className="flex items-start gap-2.5">
              <Checkbox id={id + 'images'} checked={o.images} onCheckedChange={(v) => set({ images: v === true })} className="mt-1 bg-paper" />
              <Label htmlFor={id + 'images'} className="text-[15px] font-normal leading-snug">Read text inside images with OCR. Slower on pages with pictures.</Label>
            </div>
          </fieldset>

          <fieldset className="space-y-3">
            <legend className="mb-2 font-semibold">Test attack</legend>
            <Label htmlFor={id + 'attack'} className="text-sm font-normal leading-snug text-graphite">Added to each file before it is scanned, to see the scanner respond</Label>
            <Select value={o.attack} onValueChange={(attack) => set({ attack })}>
              <SelectTrigger id={id + 'attack'} className={field}><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="none">No test attack</SelectItem>
                {health.attacks.map((a) => <SelectItem key={a.key} value={a.key}>{a.label}</SelectItem>)}
              </SelectContent>
            </Select>
            {o.attack !== 'none' && (
              <>
                <Select value={o.preset} onValueChange={(preset) => set(preset === OWN ? { preset } : { preset, payload: health.presets[preset] })}>
                  <SelectTrigger aria-label="Test instruction wording" className={field}><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {Object.keys(health.presets).map((name) => <SelectItem key={name} value={name}>{name}</SelectItem>)}
                    <SelectItem value={OWN}>Write my own</SelectItem>
                  </SelectContent>
                </Select>
                <Textarea aria-label="Test instruction text" maxLength={400} spellCheck={false} value={o.payload}
                  onChange={(e) => set({ payload: e.target.value, preset: OWN })} className="min-h-[4.5rem] bg-paper font-mono text-[13px]" />
              </>
            )}
          </fieldset>

          <fieldset className="space-y-3">
            <legend className="mb-2 font-semibold">AI screener comparison</legend>
            <div className="flex items-start gap-2.5">
              <Checkbox id={id + 'impact'} checked={o.impact} disabled={!s.available} onCheckedChange={(v) => set({ impact: v === true })} className="mt-1 bg-paper" />
              <Label htmlFor={id + 'impact'} className="text-[15px] font-normal leading-snug">Ask an AI screener to score the resume with and without the flagged text</Label>
            </div>
            {s.available ? (
              <>
                <Select value={o.screener} onValueChange={(screener) => set({ screener })}>
                  <SelectTrigger aria-label="Screener model" className={field}><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value={AUTO}>Automatic: first model that answers</SelectItem>
                    {s.options.map((x) => <SelectItem key={x.id} value={x.id}>{x.model + (x.hosted ? ', hosted by ' + x.provider : ', on this computer')}</SelectItem>)}
                  </SelectContent>
                </Select>
                <p className="text-sm text-graphite">{sentTo}</p>
              </>
            ) : (
              <p className="text-sm text-graphite">No AI screener is set up. Scanning does not need one.</p>
            )}
          </fieldset>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}

/* ------------------------------------------------------------------- report */

function DecisionBox({ title, r, note }: { title: string; r?: Decision; note?: string }) {
  const tone = r?.valid ? (r.decision === 'SELECT' ? 'text-clean' : 'text-attack') : 'text-graphite'
  return (
    <div className="min-w-0 border-t-2 border-ink pt-3">
      <h3 className="text-[15px] text-graphite">{title}</h3>
      {!r ? <p className="title mt-1 text-2xl text-graphite">No decision</p> : r.valid ? (
        <>
          <p className={cn('title mt-1 text-[28px]', tone)}>{r.decision === 'SELECT' ? 'Select' : 'Reject'}</p>
          <div className={cn('relative mb-1.5 mt-2.5 h-1.5 rounded-full bg-desk', tone)} aria-hidden="true">
            <i className="absolute inset-y-0 left-0 rounded-full bg-current" style={{ width: `${Math.max(0, Math.min(100, Number(r.score) || 0))}%` }} />
            <b className="absolute -bottom-1 -top-1 left-[60%] w-0.5 bg-ink" />
          </div>
          <p className="text-[15px]">Score {r.score} of 100</p>
          <p className="mt-1 text-sm text-graphite [overflow-wrap:anywhere]">{r.reason}</p>
        </>
      ) : (
        <>
          <p className="title mt-1 text-2xl text-graphite">Invalid answer</p>
          <p className="mt-1 text-sm text-graphite [overflow-wrap:anywhere]">{r.error}</p>
        </>
      )}
      {note && <p className="mt-1 text-sm text-graphite">{note}</p>}
    </div>
  )
}

function ImpactPanel({ report }: { report: Report }) {
  const s = report.impact
  if (!s) return null
  const a = s.results
  let line = ''
  if (s.available && a) {
    const ref = a.original, u = a.as_read, d = a.cleaned
    if (ref?.valid && u.valid && d.valid) {
      line = ref.decision !== u.decision
        ? (d.decision === ref.decision ? 'The attack changed the decision, and removing the flagged text restored it.' : 'The attack changed the decision, and removing the flagged text did not restore it.')
        : 'The attack did not change this screener’s decision.'
    } else if (u.valid && d.valid && (s.removed_blocks ?? 0) > 0) {
      line = u.decision !== d.decision ? 'Removing the flagged text changed the decision.' : 'Removing the flagged text did not change the decision.'
    }
  }
  return (
    <section className={panel}>
      <h2 className={h2}>AI screener decision</h2>
      {s.available && a ? (
        <>
          {line && <p className="max-w-[64ch]">{line}</p>}
          <div className="my-4 grid grid-cols-[repeat(auto-fit,minmax(160px,1fr))] gap-6">
            {a.original && <DecisionBox title="Original file" r={a.original} />}
            <DecisionBox title={a.original ? 'With the test attack' : 'As a program reads it'} r={a.as_read} />
            <DecisionBox title="Flagged text removed" r={a.cleaned} note={s.removed_blocks === 0 ? 'Nothing was removed, so this is the same decision.' : ''} />
          </div>
          <p className="max-w-[64ch] text-sm text-graphite">
            Scored by {s.model} through {s.provider} against a junior Python backend rubric. The line on each bar marks 60, the score needed to select.{s.skipped?.length ? ' ' + s.skipped.join(' ') : ''}
          </p>
        </>
      ) : <p className="max-w-[64ch] text-sm text-graphite">{s.reason}</p>}
    </section>
  )
}

function FindingRow({ f, number, onShow }: { f: Finding; number: number; onShow: () => void }) {
  const bits: string[] = []
  if (f.channel === 'page text' && f.agreement != null && f.hidden) {
    const saw = f.ocr_saw?.trim()
    bits.push(`OCR of the same area saw ${saw ? '“' + saw.slice(0, 90) + '”' : 'nothing'} (agreement ${Math.round(f.agreement)} of 100)`)
  }
  const scores = Object.entries(f.scores ?? {}).map(([k, x]) => `${k} ${x.toFixed(2)}`)
  if (scores.length) bits.push('instruction scores: ' + scores.join(', '))
  if (f.obfuscated) bits.push('contains invisible or look-alike characters')
  const evidence = bits.join('; ')
  return (
    <article id={`finding-${number}`} className="finding grid grid-cols-[22px_minmax(0,1fr)] gap-x-3.5 border-t border-line py-5 first-of-type:border-t-0 first-of-type:pt-1">
      <span className={cn('tag', f.severity === 'review' && 'for-review')}>{number}</span>
      <div className="min-w-0">
        <h3 className="font-semibold leading-snug">
          {f.title}
          {f.page && <span className="font-normal text-graphite"> on page {f.page}</span>}
          {f.severity === 'review' && <span className="font-normal text-graphite">, for review</span>}
        </h3>
        {f.technique_label && <p className="mt-1.5 text-[15px]"><span className="text-graphite">How it is hidden:</span> {f.technique_label}.</p>}
        {f.intents.length > 0 && <p className="mt-1.5 text-[15px]"><span className="text-graphite">What it tries to do:</span> {f.intents.join('; ')}.</p>}
        <p className={cn('extracted my-3', f.hidden && 'hidden-text')}>{isInstruction(f) ? <span className="hl text-ink">{f.text}</span> : f.text}</p>
        <p className="text-[15px]">{f.explanation}</p>
        {evidence && <p className="mt-1.5 text-[13.5px] text-graphite">{evidence.charAt(0).toUpperCase() + evidence.slice(1)}.</p>}
        {f.page && f.bbox && <Button variant="outline" size="sm" className="noprint mt-3 bg-paper" onClick={onShow}>Show on page {f.page}</Button>}
      </div>
    </article>
  )
}

function ChecksPanel({ report }: { report: Report }) {
  return (
    <section className={panel}>
      <h2 className={h2}>What was checked</h2>
      <Table className="text-sm">
        <TableBody>
          {report.checks.map((c) => (
            <TableRow key={c.name} className="hover:bg-transparent">
              <TableCell className="whitespace-normal pl-0 align-top">{c.name}</TableCell>
              <TableCell className={cn('align-top tabular-nums', c.result && 'font-semibold text-attack')}>{c.result ? `${c.result} flagged` : 'none found'}</TableCell>
              <TableCell className="whitespace-normal pr-0 align-top text-graphite">{c.detail}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <Collapsible className="mt-4">
        <CollapsibleTrigger asChild>
          <Button variant="outline" size="sm" className="noprint bg-paper">Show all {report.blocks.length} pieces of text with their scores</Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3">
          <Table className="text-sm">
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                {['Where', 'Text', 'Visible match', 'Scores'].map((x) => <TableHead key={x} className="h-9 pl-0 font-medium text-graphite">{x}</TableHead>)}
              </TableRow>
            </TableHeader>
            <TableBody>
              {report.blocks.map((b, i) => (
                <TableRow key={i} className="hover:bg-transparent">
                  <TableCell className="pl-0 align-top tabular-nums">{b.page ? 'p' + b.page + (b.channel === 'page text' ? '' : ' ' + b.channel) : b.channel}</TableCell>
                  <TableCell className={cn('whitespace-normal pl-0 align-top font-mono text-[12.5px] [overflow-wrap:anywhere]', Boolean(b.finding) && 'text-attack')}>{b.text.length > 220 ? b.text.slice(0, 220) + '…' : b.text}</TableCell>
                  <TableCell className="pl-0 align-top tabular-nums">{b.agreement == null ? '' : Math.round(b.agreement)}</TableCell>
                  <TableCell className="px-0 align-top tabular-nums">{Object.values(b.scores).map((x) => x.toFixed(2)).join(' / ')}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CollapsibleContent>
      </Collapsible>
    </section>
  )
}

function ReportSheet({ item, onShow }: { item: Item; onShow: (number: number) => void }) {
  const r = item.report
  if (!r) {
    return (
      <div className="rounded-[10px] border border-line bg-paper">
        <section className="px-6 py-8 sm:px-8" aria-live="polite">
          <p className="title text-[clamp(1.9rem,2.8vw,2.5rem)]">{STATUS[item.status]}</p>
          <p className="mt-3 max-w-[46ch] text-[17px] leading-snug text-graphite">{item.status === 'error' ? item.error : item.name}</p>
        </section>
      </div>
    )
  }
  const level = r.verdict.level
  const saveReport = () => {
    const { pages_rendered: _pages, ...copy } = r
    download(item.name.replace(/\.pdf$/i, '') + '-scan-report.json', JSON.stringify(copy, null, 2), 'application/json')
  }
  return (
    <div className={cn('overflow-hidden rounded-[10px] border border-line border-t-[3px] bg-paper', LEVEL_RULE[level])}>
      <section className="px-6 pb-7 pt-6 sm:px-8" aria-live="polite">
        <p className={cn('title flex items-center gap-3 text-[clamp(1.9rem,2.8vw,2.5rem)]', LEVEL_TEXT[level])}>
          <i className={cn('glyph size-4', level)} aria-hidden="true" />
          {LEVELS[level]}
        </p>
        <p className="mt-3 max-w-[50ch] text-[17px] leading-snug">{r.verdict.summary}</p>
        <p className="mt-2.5 text-sm text-graphite">{item.name}. {r.pages} {r.pages === 1 ? 'page' : 'pages'}, scanned in {r.seconds} seconds.</p>
        {r.test_attack && <p className="text-sm text-graphite">A test attack was added before scanning: {r.test_attack.label.toLowerCase()}.</p>}
        <div className="noprint mt-5 flex flex-wrap gap-2.5">
          <Button variant="outline" size="sm" className="bg-paper" onClick={saveReport}>Download report</Button>
          <Button variant="outline" size="sm" className="bg-paper" onClick={() => window.print()}>Print this result</Button>
        </div>
      </section>
      <ImpactPanel report={r} />
      <section className={panel}>
        <h2 className={h2}>{r.findings.length ? `${r.findings.length} ${r.findings.length === 1 ? 'finding' : 'findings'}` : 'Findings'}</h2>
        {!r.findings.length && <p className="text-[15px] text-graphite">Nothing was flagged in this file.</p>}
        {r.findings.map((f, i) => <FindingRow key={i} f={f} number={i + 1} onShow={() => onShow(i + 1)} />)}
      </section>
      <ChecksPanel report={r} />
    </div>
  )
}

/* --------------------------------------------------------------------- page */

export default function Scan() {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState('')
  const [items, setItems] = useState<Item[]>([])
  const [selected, setSelected] = useState<number | null>(null)
  const [program, setProgram] = useState(false)
  const [options, setOptions] = useState<Options>({ images: true, attack: 'none', preset: '', payload: '', impact: false, screener: AUTO })
  const itemsRef = useRef<Item[]>([])
  const optionsRef = useRef(options)
  const running = useRef(false)
  const pagesRef = useRef<HTMLElement>(null)
  optionsRef.current = options

  useEffect(() => {
    document.title = 'Scan a resume | Resume injection scanner'
    getHealth().then((h) => {
      setHealth(h)
      const first = Object.keys(h.presets)[0]
      // A link can open a result directly: /scan?sample=<file>&attack=<technique>.
      const query = new URLSearchParams(window.location.search)
      const attack = h.attacks.some((a) => a.key === query.get('attack')) ? query.get('attack')! : 'none'
      optionsRef.current = { ...optionsRef.current, preset: first, payload: h.presets[first], attack }
      setOptions(optionsRef.current)
      const sample = h.samples.find((s) => s.id === query.get('sample'))
      if (sample && !itemsRef.current.length) enqueue([{ name: sample.title + ' (sample)', sample: sample.id }])
    }).catch(() => setError('The scanner is not running. Start it with: python scanner/scanner_app.py'))
  }, [])

  const update = (id: number, patch: Partial<Item>) => {
    itemsRef.current = itemsRef.current.map((i) => (i.id === id ? { ...i, ...patch } : i))
    setItems(itemsRef.current)
  }

  const run = async () => {
    if (running.current) return
    running.current = true
    for (;;) {
      const item = itemsRef.current.find((i) => i.status === 'waiting')
      if (!item) break
      update(item.id, { status: 'scanning' })
      setSelected((s) => s ?? item.id)
      const o = optionsRef.current
      try {
        const report = await scanFile(item.name, { file: item.file, sample: item.sample }, { ...o, screener: o.screener === AUTO ? '' : o.screener })
        update(item.id, { status: 'done', report, file: undefined })
      } catch (err) {
        update(item.id, { status: 'error', error: err instanceof Error ? err.message : 'The scan failed.', file: undefined })
      }
    }
    running.current = false
  }

  const enqueue = (entries: Pick<Item, 'name' | 'file' | 'sample'>[]) => {
    setError('')
    const start = itemsRef.current.length
    itemsRef.current = [...itemsRef.current, ...entries.map((e, k) => ({ ...e, id: start + k, status: 'waiting' as const }))]
    setItems(itemsRef.current)
    void run()
  }

  const addFiles = useCallback((files: File[]) => {
    const pdfs = files.filter((f) => /\.pdf$/i.test(f.name) || f.type === 'application/pdf')
    if (!pdfs.length) return setError('Only PDF files can be scanned.')
    enqueue(pdfs.map((file) => ({ name: file.name, file })))
  }, [])

  const current = items.find((i) => i.id === selected)
  const report = current?.report
  const markCount = report ? report.findings.filter((f) => f.page && f.bbox).length : 0

  const showMark = (number: number, smooth: boolean) => {
    setProgram(true)
    requestAnimationFrame(() => {
      const mark = document.getElementById(`mark-${number}`), col = pagesRef.current
      if (!mark || !col) return
      const sheet = mark.closest<HTMLElement>('.sheet')!
      if (getComputedStyle(col).position === 'sticky') col.scrollTo({ top: Math.max(0, sheet.offsetTop + mark.offsetTop - col.clientHeight / 3), behavior: smooth ? 'smooth' : 'auto' })
      else if (smooth) mark.scrollIntoView({ block: 'center', behavior: 'smooth' })
      mark.classList.remove('pulse'); void mark.offsetWidth; mark.classList.add('pulse')
    })
  }
  const showFinding = (number: number) => {
    const row = document.getElementById(`finding-${number}`)
    if (!row) return
    row.scrollIntoView({ block: 'center', behavior: 'smooth' })
    row.classList.remove('pulse'); void row.offsetWidth; row.classList.add('pulse')
  }

  // A new result opens in the program's reading when it has something to show, at its first mark.
  useEffect(() => {
    setProgram(markCount > 0)
    if (!report || !markCount) return
    const first = report.findings.findIndex((f) => f.page && f.bbox) + 1
    const images = [...(pagesRef.current?.querySelectorAll('img') ?? [])]
    let live = true
    Promise.all(images.map((i) => i.decode().catch(() => undefined))).then(() => { if (live) showMark(first, false) })
    return () => { live = false }
  }, [report])

  const done = items.filter((i) => i.report)
  const count = (l: Level) => done.filter((i) => i.report!.verdict.level === l).length
  const pending = items.filter((i) => i.status === 'waiting' || i.status === 'scanning').length
  const saveCsv = () => {
    const q = (s: unknown) => '"' + String(s).replace(/"/g, '""') + '"'
    const rows: unknown[][] = [['file', 'verdict', 'attack_types', 'review_types', 'findings', 'pages', 'seconds', 'error']]
    for (const i of items) {
      if (i.report) rows.push([i.name, i.report.verdict.level, i.report.verdict.attack_types.join('; '), i.report.verdict.review_types.join('; '), i.report.findings.length, i.report.pages, i.report.seconds, ''])
      else if (i.status === 'error') rows.push([i.name, 'error', '', '', '', '', '', i.error])
    }
    download('scan-summary.csv', rows.map((r) => r.map(q).join(',')).join('\n'), 'text/csv')
  }
  const working = items.length > 0

  return (
    <div className="min-h-svh">
      <Nav page="scan" wide />
      <main className="mx-auto max-w-[1520px] px-5 pb-16 pt-10 md:px-8">
        <div className={cn('noprint grid gap-x-14 gap-y-6', working ? 'lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-center' : 'mx-auto max-w-[1176px] pt-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)] lg:items-center')}>
          <div className={cn(working && 'lg:hidden')}>
            <h1 className={cn(working ? 'title text-2xl' : 'display text-[clamp(2.5rem,4.2vw,3.5rem)]')}>Scan a resume</h1>
            {!working && (
              <p className="mt-5 max-w-[42ch] text-[19px] leading-relaxed text-graphite">
                Drop one PDF or a batch. Each gets a verdict, the findings by name, and its pages with the hidden text marked. Scanning runs on this computer and files are not stored.
              </p>
            )}
          </div>
          <Dropzone compact={working} accept="application/pdf,.pdf" onFiles={addFiles}
            title={working ? 'Drop more resume PDFs here' : 'Drop resume PDFs here'}
            description={working ? undefined : 'One file or many. Each is scanned in turn.'} action="Choose files" />
          {health && health.samples.length > 0 && (
            <div className={cn('flex flex-wrap items-center gap-3', !working && 'lg:col-start-2')}>
              <Label htmlFor="sample" className="text-[15px] font-normal text-graphite">No resume to hand?</Label>
              <Select value="" onValueChange={(id) => enqueue([{ name: health.samples.find((s) => s.id === id)!.title + ' (sample)', sample: id }])}>
                <SelectTrigger id="sample" className={cn(field, 'min-w-60 flex-1')}><SelectValue placeholder="Scan a sample from the experiment…" /></SelectTrigger>
                <SelectContent>
                  {[...new Set(health.samples.map((s) => s.group))].map((group) => (
                    <SelectGroup key={group}>
                      <SelectLabel>{group}</SelectLabel>
                      {health.samples.filter((s) => s.group === group).map((s) => <SelectItem key={s.id} value={s.id}>{s.title}</SelectItem>)}
                    </SelectGroup>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>
        {error && <p role="alert" className="mx-auto mt-5 max-w-[1176px] font-medium text-attack">{error}</p>}
        <div className={cn(!working && 'mx-auto max-w-[1176px]')}>
          {health && <OptionsPanel health={health} o={options} set={(patch) => setOptions((o) => ({ ...o, ...patch }))} />}
        </div>

        {working && (
          <div className="mt-7 grid items-start gap-6 lg:grid-cols-2 xl:grid-cols-[232px_minmax(0,1fr)_minmax(0,1.08fr)] xl:gap-8">
            <nav aria-label="Scanned files" className="noprint lg:col-span-2 xl:sticky xl:top-4 xl:col-span-1 xl:max-h-[calc(100svh-2rem)] xl:overflow-auto xl:p-0.5">
              <p role="status" className="mb-3 text-sm text-graphite">
                {items.length} {items.length === 1 ? 'file' : 'files'}: {count('attack')} with an attack, {count('review')} to review, {count('clean')} clean{pending ? `, ${pending} still to scan` : ''}.
              </p>
              <div className="grid grid-cols-[repeat(auto-fill,minmax(210px,1fr))] items-start gap-x-2 xl:block">
                {items.map((item) => {
                  const level = item.report?.verdict.level
                  const types = item.report ? (item.report.verdict.attack_types.length ? item.report.verdict.attack_types : item.report.verdict.review_types).join(', ') : item.error
                  return (
                    <button key={item.id} type="button" aria-current={item.id === selected || undefined} onClick={() => setSelected(item.id)}
                      className={cn('mb-1 grid w-full grid-cols-[12px_minmax(0,1fr)] gap-x-3 rounded-md px-3 py-2.5 text-left transition-colors', item.id === selected ? 'bg-desk' : 'hover:bg-desk/60')}>
                      <i className={cn('glyph mt-[5px]', level ?? item.status)} aria-hidden="true" />
                      <span className="min-w-0">
                        <span className={cn('block text-[15px] font-semibold', level && LEVEL_TEXT[level])}>{level ? LEVELS[level] : STATUS[item.status]}</span>
                        <span className="block text-sm [overflow-wrap:anywhere]">{item.name}</span>
                        {types && <span className="block text-[13px] text-graphite">{types}</span>}
                      </span>
                    </button>
                  )
                })}
              </div>
              {done.length > 0 && <Button variant="outline" size="sm" className="mt-3 bg-paper" onClick={saveCsv}>Download summary (CSV)</Button>}
            </nav>

            <div className="min-w-0">{current && <ReportSheet item={current} onShow={(n) => showMark(n, true)} />}</div>

            {report && (
              <section ref={pagesRef} aria-label="Rendered pages" className="stage min-w-0 px-5 pb-1 lg:sticky lg:top-4 lg:max-h-[calc(100svh-2rem)] lg:overflow-auto">
                <div className="noprint z-[3] -mx-5 rounded-t-[10px] bg-desk px-5 pb-4 pt-4 lg:sticky lg:top-0">
                  <Segmented label="Page view" value={program ? 'program' : 'person'} onChange={(v) => setProgram(v === 'program')}
                    options={[{ id: 'person', label: 'As a person sees it' }, { id: 'program', label: 'As a program reads it' }]} />
                  {program ? (
                    <p className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-sm text-graphite">
                      <span className="flex items-center gap-2"><i className="h-3 w-6 rounded-[2px] bg-select-tint shadow-[0_0_0_1.5px_var(--select)]" />In the file but not printed</span>
                      <span className="flex items-center gap-2"><i className="h-3 w-6 rounded-[2px] bg-marker" />Reads as an instruction</span>
                    </p>
                  ) : <p className="mt-3 text-sm text-graphite">The page as it displays and prints. Nothing is marked.</p>}
                </div>
                {report.pages_rendered.map((page) => (
                  <div key={page.number}>
                    <Sheet image={page.image} alt={`Page ${page.number} as a reader sees it`} page={page} program={program} onMark={showFinding} markId={(n) => `mark-${n}`}
                      marks={report.findings.map((finding, i) => ({ finding, number: i + 1 })).filter((m) => m.finding.page === page.number && m.finding.bbox)} />
                    <p className="mb-5 mt-2.5 text-[13px] text-graphite">Page {page.number}</p>
                  </div>
                ))}
              </section>
            )}
          </div>
        )}

        {health && (
          <footer className={cn('mt-14 space-y-2.5 border-t border-line pt-5 text-sm text-graphite', !working && 'mx-auto max-w-[1176px]')}>
            <p className="max-w-[84ch]">Limits: {health.limits.join(' ')}</p>
            <p className="noprint max-w-[84ch]">
              Scanner version {health.version}. Detectors: {health.detectors.map((d) => `${d.name} (threshold ${d.threshold})`).join(', ')}. Files of up to {health.max_pages} pages and 20 MB.
            </p>
          </footer>
        )}
      </main>
    </div>
  )
}
