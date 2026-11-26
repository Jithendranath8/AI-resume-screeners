import { type DragEvent, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface DropzoneProps {
  onFiles: (files: File[]) => void
  accept: string
  title: string
  description?: string
  action: string
  /** A single row, used once files are already being scanned. */
  compact?: boolean
  className?: string
}

/** Where files are dropped or chosen. */
export function Dropzone({ onFiles, accept, title, description, action, compact, className }: DropzoneProps) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const drop = (e: DragEvent) => {
    e.preventDefault()
    setOver(false)
    onFiles([...e.dataTransfer.files])
  }
  return (
    <div onDragOver={(e) => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)} onDrop={drop}
      className={cn('rounded-[10px] border-[1.5px] border-dashed transition-colors', over ? 'border-select bg-select-tint/50' : 'border-line-strong bg-paper',
        compact ? 'flex flex-wrap items-center justify-between gap-x-6 gap-y-3 px-5 py-3.5' : 'px-8 py-10 text-center', className)}>
      <div>
        <p className={cn('font-medium', compact ? 'text-[15px]' : 'text-lg')}>{title}</p>
        {description && <p className="mt-1 text-[15px] text-graphite">{description}</p>}
      </div>
      <Button type="button" variant={compact ? 'outline' : 'default'} className={cn('font-medium', compact ? 'h-9 px-4' : 'mt-5 h-10 px-5 text-[15px]')} onClick={() => input.current?.click()}>{action}</Button>
      <input ref={input} type="file" accept={accept} multiple hidden onChange={(e) => { onFiles([...(e.target.files ?? [])]); e.target.value = '' }} />
    </div>
  )
}
