import { Link } from 'react-router-dom'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

const SECTIONS = [
  ['#attacks', 'Attack types'],
  ['#method', 'How it decides'],
  ['#results', 'Results'],
]

/** The mark: a page with one selected line that has nothing printed under it. */
export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 20 24" className={cn('h-6 w-5', className)} aria-hidden="true">
      <rect x="0.75" y="0.75" width="18.5" height="22.5" rx="2" fill="#fff" stroke="currentColor" strokeWidth="1.5" />
      <path d="M5 6.5h10M5 10h7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <rect x="4" y="13.5" width="12" height="4" rx="0.75" fill="var(--select)" />
    </svg>
  )
}

/** Site header, the same on both pages. */
export function Nav({ page, wide }: { page: 'home' | 'scan'; wide?: boolean }) {
  const link = 'rounded-sm py-1 text-[15px] text-graphite hover:text-ink'
  return (
    <header className="noprint border-b border-line">
      <div className={cn('mx-auto flex h-16 w-full items-center justify-between gap-8 px-5 md:px-8', wide ? 'max-w-[1520px]' : 'max-w-[1240px]')}>
        <Link to="/" className="flex items-center gap-2.5 whitespace-nowrap rounded-sm text-[15px] font-semibold tracking-[-0.005em] text-ink">
          <Logo />
          Resume injection scanner
        </Link>
        <nav aria-label="Main" className="flex items-center gap-7 whitespace-nowrap">
          {page === 'home' ? (
            <>
              {SECTIONS.map(([href, label]) => <a key={href} href={href} className={cn(link, 'hidden lg:inline')}>{label}</a>)}
              <Button asChild size="sm" className="h-9 px-4 text-[15px] font-medium"><Link to="/scan">Scan a resume</Link></Button>
            </>
          ) : (
            <>
              <Link to="/" className={cn(link, 'max-sm:hidden')}>About the scanner</Link>
              <span aria-current="page" className="py-1 text-[15px] font-medium text-ink">Scan a resume</span>
            </>
          )}
        </nav>
      </div>
    </header>
  )
}
