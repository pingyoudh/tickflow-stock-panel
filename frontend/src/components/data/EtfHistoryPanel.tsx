import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Loader2 } from 'lucide-react'
import { api } from '@/lib/api'
import { QK } from '@/lib/queryKeys'

export function EtfHistoryPanel({ caps, isRunning, earliestDate, onStarted }: {
  caps: { label: string; capabilities: Record<string, { rpm: number | null; batch: number | null; subscribe: number | null }> } | undefined
  isRunning: boolean
  earliestDate: string | null
  onStarted: (jobId: string) => void
}) {
  const qc = useQueryClient()
  const [value, setValue] = useState(4)
  const [unit, setUnit] = useState<'month' | 'year'>('year')
  const hasBatchCap = !!caps?.capabilities?.['kline.daily.batch']
  const offsetDays = unit === 'month' ? value * 30 : value * 365
  const baseDate = earliestDate ? new Date(earliestDate) : new Date()
  const targetDate = new Date(baseDate)
  targetDate.setDate(targetDate.getDate() - offsetDays)
  const targetDateText = targetDate.toISOString().slice(0, 10)

  const extend = useMutation({
    mutationFn: () => api.extendEtfHistory(value, unit),
    onSuccess: ({ job_id }) => {
      onStarted(job_id)
      qc.invalidateQueries({ queryKey: QK.pipelineJobs })
    },
  })

  return (
    <div className="space-y-3">
      <div className="rounded-card border border-accent/20 bg-accent/8 p-3">
        <div className="text-xs text-foreground">向前扩展 ETF 历史</div>
        <div className="mt-1 text-[10px] leading-relaxed text-muted">
          同步 ETF 维表、复权因子和日 K，并重算目标区间技术指标。默认再向前补 4 年，可满足多数日频模型的基础历史长度。
        </div>
      </div>

      <div className="flex items-center justify-between gap-3">
        <span className="text-xs text-secondary">扩展跨度</span>
        <div className="flex items-center gap-2">
          <div className="flex items-center">
            <button
              onClick={() => setValue(Math.max(1, value - 1))}
              disabled={!hasBatchCap || isRunning}
              className="flex h-7 w-7 items-center justify-center rounded-l-btn border border-border bg-elevated text-xs text-secondary hover:bg-border/50 disabled:opacity-30"
            >−</button>
            <div className="flex h-7 w-9 items-center justify-center border-y border-border bg-base font-mono text-xs tabular-nums text-foreground">
              {value}
            </div>
            <button
              onClick={() => setValue(Math.min(unit === 'year' ? 10 : 36, value + 1))}
              disabled={!hasBatchCap || isRunning}
              className="flex h-7 w-7 items-center justify-center rounded-r-btn border border-border bg-elevated text-xs text-secondary hover:bg-border/50 disabled:opacity-30"
            >+</button>
          </div>
          <div className="flex overflow-hidden rounded-btn border border-border">
            {(['month', 'year'] as const).map(item => (
              <button
                key={item}
                onClick={() => {
                  setUnit(item)
                  if (item === 'year' && value > 10) setValue(4)
                }}
                disabled={isRunning}
                className={`h-7 px-2 text-[10px] font-medium ${unit === item ? 'bg-accent/15 text-accent' : 'text-secondary hover:bg-elevated'}`}
              >{item === 'month' ? '月' : '年'}</button>
            ))}
          </div>
        </div>
      </div>

      <div className="space-y-1 border-y border-border py-2 text-[10px]">
        <div className="flex justify-between"><span className="text-muted">当前最早</span><span className="font-mono text-secondary">{earliestDate ?? '暂无本地数据'}</span></div>
        <div className="flex justify-between"><span className="text-muted">目标起点</span><span className="font-mono text-secondary">{targetDateText}</span></div>
      </div>

      <button
        onClick={() => extend.mutate()}
        disabled={!hasBatchCap || isRunning || extend.isPending}
        className="inline-flex h-8 w-full items-center justify-center gap-1.5 rounded-btn bg-accent text-xs font-medium text-white disabled:pointer-events-none disabled:opacity-40"
      >
        {extend.isPending ? <><Loader2 className="h-3.5 w-3.5 animate-spin" />正在创建任务</> : '获取 ETF 历史'}
      </button>

      {extend.isError && <div className="text-[10px] text-danger">{String(extend.error.message)}</div>}
      {!hasBatchCap && <div className="text-center text-[10px] text-warning">需要批量日 K 权限</div>}
      <div className="text-[10px] leading-relaxed text-muted">多年全市场数据量较大，任务会在后台运行，进度可在本页顶部和同步历史中查看。</div>
    </div>
  )
}
