import { type KeyboardEvent, useRef } from 'react'

import { cn } from '@/lib/utils'

interface SegmentedProps<T extends string> {
  label: string
  value: T
  onChange: (value: T) => void
  options: { id: T; label: string }[]
  className?: string
}

/** A two- or three-way switch. One option is always chosen; arrow keys move between them. */
export function Segmented<T extends string>({ label, value, onChange, options, className }: SegmentedProps<T>) {
  const ref = useRef<HTMLDivElement>(null)
  const move = (e: KeyboardEvent) => {
    const step = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0
    if (!step) return
    e.preventDefault()
    const next = options[(options.findIndex((o) => o.id === value) + step + options.length) % options.length]
    onChange(next.id)
    ref.current?.querySelector<HTMLElement>(`[data-id="${next.id}"]`)?.focus()
  }
  return (
    <div ref={ref} role="radiogroup" aria-label={label} onKeyDown={move} className={cn('inline-flex rounded-md bg-ink/[0.07] p-[3px]', className)}>
      {options.map((o) => (
        <button key={o.id} type="button" role="radio" data-id={o.id} aria-checked={o.id === value} tabIndex={o.id === value ? 0 : -1} onClick={() => onChange(o.id)}
          className={cn('rounded-[4px] px-3 py-1.5 text-sm leading-tight transition-colors', o.id === value ? 'bg-paper font-medium text-ink shadow-[0_0_0_1px_rgb(17_20_23/0.08),0_1px_2px_rgb(17_20_23/0.12)]' : 'text-graphite hover:text-ink')}>
          {o.label}
        </button>
      ))}
    </div>
  )
}
