import { type CSSProperties, useLayoutEffect, useRef } from 'react'

import { cn } from '@/lib/utils'
import { type Finding, type PageSize, isInstruction, markBox } from '@/lib/scanner'

export interface SheetMark { finding: Finding; number: number }

interface SheetProps {
  image: string
  alt: string
  /** Size of the area the image shows, in the PDF's own units. */
  page: PageSize
  /** Offset of that area from the top of the PDF page, for a cropped band. */
  top?: number
  marks: SheetMark[]
  /** Show the page as a program reads it: every line of text selected, and what is not printed spelled out. */
  program: boolean
  /** Boxes of every line of text in the file, in the PDF's own units. */
  lines?: number[][]
  /** Bring the selection in line by line, top to bottom, instead of all at once. */
  sweep?: boolean
  /** Show each mark's finding number. Off where there is no numbered list beside the page. */
  tags?: boolean
  onMark?: (number: number) => void
  markId?: (number: number) => string
  className?: string
}

/** Marks show their text, so two that sit close together would overlap. Move each one clear of those above it. */
function separate(sheet: HTMLElement) {
  const marks = [...sheet.querySelectorAll<HTMLElement>('.mark')]
  for (const m of marks) m.style.top = m.dataset.top ?? ''
  const base = sheet.getBoundingClientRect()
  const placed: { left: number; right: number; top: number; bottom: number }[] = []
  const boxes = marks.map((m) => ({ m, r: m.getBoundingClientRect() })).sort((a, b) => a.r.top - b.r.top)
  for (const { m, r } of boxes) {
    let top = r.top
    for (let moved = true; moved;) {
      moved = false
      for (const p of placed) {
        if (r.left < p.right && r.right > p.left && top < p.bottom + 4 && top + r.height > p.top) { top = p.bottom + 4; moved = true }
      }
    }
    if (top !== r.top) m.style.top = `${top - base.top}px`
    placed.push({ left: r.left, right: r.right, top, bottom: top + r.height })
  }
}

/** A rendered page. In the program view, text that is in the file but not on the page shows where it sits. */
export function Sheet({ image, alt, page, top = 0, marks, program, lines, sweep, tags = true, onMark, markId, className }: SheetProps) {
  const ref = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const sheet = ref.current
    if (!sheet) return
    separate(sheet)
    const watch = new ResizeObserver(() => separate(sheet))
    watch.observe(sheet)
    return () => watch.disconnect()
  }, [marks.length, image])

  return (
    <div ref={ref} className={cn('sheet', program && 'program', sweep && 'sweep', className)}>
      <img src={image} alt={alt} draggable={false} />
      {lines?.map(([x0, y0, x1, y1], i) => (
        <i key={i} className="sel" style={{ '--i': i, left: `${(100 * x0) / page.width}%`, top: `${(100 * (y0 - top)) / page.height}%`,
          width: `${(100 * (x1 - x0)) / page.width}%`, height: `${Math.max(0.5, (100 * (y1 - y0)) / page.height)}%` } as CSSProperties} />
      ))}
      {marks.map(({ finding: f, number }) => {
        if (!f.bbox) return null
        const [x0, y0, x1, y1] = f.bbox
        const box = markBox({ ...f, bbox: [x0, y0 - top, x1, y1 - top] }, page)
        const cls = cn('mark', f.hidden && 'hidden-text', isInstruction(f) && 'instruction', f.severity === 'review' && 'for-review')
        const content = (
          <>
            {tags && <span className={cn('tag', f.severity === 'review' && 'for-review')}>{number}</span>}
            {f.hidden && box.outside && 'outside the page: '}
            {f.hidden && (isInstruction(f) ? <span className="hl">{f.text}</span> : f.text)}
          </>
        )
        return onMark ? (
          <button key={number} type="button" id={markId?.(number)} className={cls} style={box.style} data-top={box.style.top} title={f.title}
            aria-label={`Finding ${number}: ${f.title}, show details`} onClick={() => onMark(number)}>
            {content}
          </button>
        ) : (
          <div key={number} className={cls} style={box.style} data-top={box.style.top}>{content}</div>
        )
      })}
    </div>
  )
}
