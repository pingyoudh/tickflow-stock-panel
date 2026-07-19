import { useState } from 'react'
import { ChevronDown, FlaskConical, ServerCog, ShieldCheck } from 'lucide-react'
import type { DataDimensionStatus } from '@/lib/api'
import { SectionTitle } from './SectionTitle'

function Metric({ dimension }: { dimension: DataDimensionStatus }) {
  return (
    <div className="flex shrink-0 items-center gap-4 text-[11px]">
      <span className="hidden sm:inline font-mono text-secondary">
        {dimension.records == null ? '—' : dimension.records.toLocaleString()} 记录
      </span>
      <span className="font-mono text-secondary">{dimension.files} 文件</span>
      <span className="w-20 text-right font-mono text-muted">
        {dimension.size_mb.toFixed(2)} MB
      </span>
    </div>
  )
}

function AssetRow({
  dimension,
  nested = false,
}: {
  dimension: DataDimensionStatus
  nested?: boolean
}) {
  const [open, setOpen] = useState(false)
  const expandable = dimension.children.length > 0

  return (
    <div className={nested ? 'bg-base/20' : ''}>
      <button
        type="button"
        onClick={() => expandable && setOpen(value => !value)}
        className={`flex min-h-10 w-full items-center justify-between gap-3 px-4 text-left ${
          expandable ? 'hover:bg-elevated/50' : 'cursor-default'
        }`}
        aria-expanded={expandable ? open : undefined}
      >
        <div className="flex min-w-0 items-center gap-2">
          {expandable ? (
            <ChevronDown
              className={`h-3.5 w-3.5 shrink-0 text-muted transition-transform ${
                open ? '' : '-rotate-90'
              }`}
            />
          ) : (
            <span className="ml-1 h-1.5 w-1.5 shrink-0 rounded-full bg-border" />
          )}
          <span className={`${nested ? 'text-[11px] text-secondary' : 'text-xs text-foreground'}`}>
            {dimension.label}
          </span>
          {dimension.sensitive && (
            <span title="敏感资产仅展示汇总">
              <ShieldCheck className="h-3.5 w-3.5 text-warning" />
            </span>
          )}
        </div>
        <Metric dimension={dimension} />
      </button>
      {expandable && open && (
        <div className="border-t border-border/50">
          {dimension.children.map(child => (
            <AssetRow key={child.id} dimension={child} nested />
          ))}
        </div>
      )}
    </div>
  )
}

function AssetSection({
  title,
  dimensions,
  icon,
}: {
  title: string
  dimensions: DataDimensionStatus[]
  icon: React.ComponentType<{ className?: string }>
}) {
  const [open, setOpen] = useState(false)
  const files = dimensions.reduce((sum, item) => sum + item.files, 0)
  const size = dimensions.reduce((sum, item) => sum + item.size_mb, 0)

  return (
    <section>
      <button
        type="button"
        onClick={() => setOpen(value => !value)}
        className="flex w-full items-center justify-between gap-3 py-2 text-left"
        aria-expanded={open}
      >
        <SectionTitle icon={icon}>
          {title}
        </SectionTitle>
        <div className="flex items-center gap-3 text-[11px]">
          <span className="font-mono text-secondary">{files} 文件</span>
          <span className="w-20 text-right font-mono text-muted">{size.toFixed(2)} MB</span>
          <ChevronDown
            className={`h-4 w-4 text-muted transition-transform ${open ? '' : '-rotate-90'}`}
          />
        </div>
      </button>
      {open && (
        <div className="divide-y divide-border overflow-hidden rounded-card border border-border bg-surface">
          {dimensions.map(dimension => (
            <AssetRow key={dimension.id} dimension={dimension} />
          ))}
        </div>
      )}
    </section>
  )
}

export function PersistentAssetSections({
  dimensions,
}: {
  dimensions: DataDimensionStatus[]
}) {
  const research = dimensions.filter(item => item.category === 'research')
  const system = dimensions.filter(item => item.category === 'system')

  return (
    <div className="space-y-2">
      <AssetSection title="用户与研究" dimensions={research} icon={FlaskConical} />
      <AssetSection title="系统资产" dimensions={system} icon={ServerCog} />
    </div>
  )
}
