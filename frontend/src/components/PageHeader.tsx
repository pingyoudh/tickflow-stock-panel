import { cn } from '@/lib/cn'

interface Props {
  title: string
  subtitle?: string
  /** 标题右侧、subtitle 之前的额外节点(如状态徽标) */
  titleExtra?: React.ReactNode
  right?: React.ReactNode
  className?: string
}

export function PageHeader({ title, subtitle, titleExtra, right, className }: Props) {
  return (
    <header
      className={cn(
        'flex flex-col items-stretch gap-2 border-b border-border px-3 pb-2 pt-3 sm:flex-row sm:items-center sm:justify-between sm:px-5',
        className,
      )}
    >
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <h1 className="shrink-0 text-lg font-semibold">{title}</h1>
        {titleExtra}
        {subtitle && <span className="text-xs text-muted">{subtitle}</span>}
      </div>
      {right && <div className="min-w-0 max-w-full sm:ml-auto">{right}</div>}
    </header>
  )
}
