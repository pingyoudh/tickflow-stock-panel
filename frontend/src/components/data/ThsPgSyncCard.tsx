import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  KeyRound,
  Loader2,
  Play,
  RefreshCw,
  ShieldCheck,
  XCircle,
} from 'lucide-react'
import { api, type PipelineJob, type ThsPgGapItem } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { formatNumber } from '@/lib/format'

function fmtTime(value?: string | null) {
  if (!value) return '—'
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return value
  return d.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
}

function statusBadge(item: ThsPgGapItem) {
  switch (item.status) {
    case 'covered':
      return { label: '已有', cls: 'text-secondary bg-elevated' }
    case 'missing':
      return { label: '缺少', cls: 'text-warning bg-warning/8' }
    case 'snapshot_only':
      return { label: '仅快照', cls: 'text-warning bg-warning/8' }
    case 'not_usable':
      return { label: '不可用', cls: 'text-muted bg-elevated' }
    case 'deferred_heavy':
      return { label: '暂缓', cls: 'text-muted bg-elevated' }
  }
}

function rowsText(item: ThsPgGapItem) {
  const local = item.local?.rows
  const source = item.source?.rows
  if (source != null) return `${formatNumber(local ?? 0)} / ${formatNumber(source)}`
  return local != null ? formatNumber(local) : '—'
}

export function ThsPgSyncCard({
  onJobStarted,
  disabled,
  job,
}: {
  onJobStarted: (jobId: string) => void
  disabled?: boolean
  job?: PipelineJob
}) {
  const qc = useQueryClient()
  const [editing, setEditing] = useState(false)
  const [dsn, setDsn] = useState('')

  const status = useQuery({
    queryKey: QK.thsPgStatus,
    queryFn: api.thsPgStatus,
    staleTime: 30_000,
  })

  const gaps = useQuery({
    queryKey: QK.thsPgGaps,
    queryFn: api.thsPgGaps,
    refetchOnWindowFocus: false,
  })

  const save = useMutation({
    mutationFn: (url: string | null) => api.thsPgSaveConfig(url),
    onSuccess: () => {
      setDsn('')
      setEditing(false)
      qc.invalidateQueries({ queryKey: QK.thsPgStatus })
      qc.invalidateQueries({ queryKey: QK.thsPgGaps })
    },
  })

  const sync = useMutation({
    mutationFn: api.thsPgSync,
    onSuccess: ({ job_id }) => {
      onJobStarted(job_id)
      qc.invalidateQueries({ queryKey: QK.pipelineJobs })
    },
  })

  const configured = status.data?.configured ?? gaps.data?.configured ?? false
  const readonlyOk = gaps.data?.readonly_ok ?? false
  const error = gaps.data?.error
  const items = gaps.data?.items ?? []
  const recommended = useMemo(() => items.filter(item => item.recommended), [items])
  const latestState = status.data?.state?.datasets ?? {}
  const canSync = configured && !error && !disabled && !sync.isPending
  const isThsJob = Boolean(
    job && (
      job.stage.startsWith('ths_pg_')
      || job.result?.financial_metrics
      || job.result?.block_membership
      || job.result?.st_status
    ),
  )
  const latestJobMessage = job?.error || job?.log.at(-1)?.msg || '等待同步任务启动'
  const jobStatusLabel = job?.status === 'failed'
    ? '同步失败'
    : job?.status === 'succeeded'
      ? '同步完成'
      : '同步中'

  const statusMeta = !configured
    ? { icon: KeyRound, label: '未配置', cls: 'text-muted bg-elevated' }
    : error
      ? { icon: XCircle, label: '只读预检失败', cls: 'text-danger bg-danger/8' }
      : readonlyOk
        ? { icon: ShieldCheck, label: '只读已确认', cls: 'text-accent bg-accent/8' }
        : { icon: AlertTriangle, label: '待审计', cls: 'text-warning bg-warning/8' }
  const StatusIcon = statusMeta.icon

  return (
    <div className="rounded-card border border-border bg-surface p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-secondary" />
            <h3 className="text-sm font-medium text-foreground">THS PG 数据缺口</h3>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            <span className={`inline-flex items-center gap-1 rounded px-1.5 py-px text-[10px] font-medium ${statusMeta.cls}`}>
              <StatusIcon className="h-3 w-3" />
              {statusMeta.label}
            </span>
            <span className="rounded bg-elevated px-1.5 py-px text-[10px] font-medium text-secondary">
              外库严格只读
            </span>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <button
            type="button"
            onClick={() => gaps.refetch()}
            disabled={gaps.isFetching}
            className="p-1 rounded hover:bg-elevated text-secondary hover:text-accent disabled:opacity-50 transition-colors"
            title="重新审计"
          >
            {gaps.isFetching ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          </button>
          <button
            type="button"
            onClick={() => sync.mutate()}
            disabled={!canSync}
            className="inline-flex items-center gap-1 rounded-btn border border-accent/30 bg-accent/10 px-2 py-1 text-[11px] font-medium text-accent hover:bg-accent/15 disabled:opacity-40 disabled:pointer-events-none transition-colors"
          >
            {sync.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            同步推荐
          </button>
        </div>
      </div>

      <div className="mt-3 space-y-2">
        <div className="flex items-center justify-between gap-3 text-[11px]">
          <span className="text-muted">连接</span>
          <span className="truncate font-mono text-secondary">
            {status.data?.masked_dsn || '—'}
          </span>
        </div>
        <div className="flex items-center justify-between gap-3 text-[11px]">
          <span className="text-muted">最近同步</span>
          <span className="font-mono text-secondary">
            {fmtTime(
              latestState.financial_metrics?.last_success_at
              || latestState.block_membership?.last_success_at
              || latestState.st_status?.last_success_at,
            )}
          </span>
        </div>
      </div>

      {(editing || !configured) && (
        <div className="mt-3 flex items-center gap-2">
          <input
            type="password"
            value={dsn}
            onChange={(e) => setDsn(e.target.value)}
            placeholder="postgresql://..."
            className="min-w-0 flex-1 rounded-btn border border-border bg-base px-2 py-1.5 font-mono text-xs text-foreground outline-none focus:border-accent/60"
          />
          <button
            type="button"
            onClick={() => save.mutate(dsn.trim() || null)}
            disabled={save.isPending}
            className="rounded-btn bg-accent/15 px-2.5 py-1.5 text-xs font-medium text-accent disabled:opacity-50"
          >
            保存
          </button>
        </div>
      )}

      {configured && !editing && (
        <div className="mt-3 flex items-center gap-2 text-[11px]">
          <button
            type="button"
            onClick={() => setEditing(true)}
            className="text-secondary hover:text-accent transition-colors"
          >
            更新连接
          </button>
          <span className="text-border">·</span>
          <button
            type="button"
            onClick={() => save.mutate(null)}
            disabled={save.isPending}
            className="text-muted hover:text-danger disabled:opacity-50 transition-colors"
          >
            清除连接
          </button>
        </div>
      )}

      {error && (
        <div className="mt-3 rounded-btn border border-danger/35 bg-danger/5 px-3 py-2 text-xs text-danger">
          {error}
        </div>
      )}

      {isThsJob && job && (
        <div className="mt-3 border-y border-border/70 py-2.5">
          <div className="flex items-center justify-between gap-3 text-[11px]">
            <span className={job.status === 'failed' ? 'text-danger' : 'text-secondary'}>
              {jobStatusLabel}
            </span>
            <span className="font-mono text-foreground">{job.progress}%</span>
          </div>
          <div className="mt-1.5 h-1 overflow-hidden bg-elevated">
            <div
              className={`h-full transition-[width] duration-300 ${
                job.status === 'failed' ? 'bg-danger' : 'bg-accent'
              }`}
              style={{ width: `${job.progress}%` }}
            />
          </div>
          <p className={`mt-1.5 truncate text-[10px] ${
            job.status === 'failed' ? 'text-danger' : 'text-muted'
          }`} title={latestJobMessage}>
            {latestJobMessage}
          </p>
        </div>
      )}

      <div className="mt-4 space-y-1.5">
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-muted">推荐补缺</span>
          <span className="font-mono text-secondary">{recommended.length} 项</span>
        </div>
        <div className="max-h-52 overflow-y-auto rounded-btn border border-border bg-base/30">
          {gaps.isLoading ? (
            <div className="flex items-center gap-2 px-3 py-3 text-xs text-muted">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              审计中…
            </div>
          ) : items.length === 0 ? (
            <div className="px-3 py-3 text-xs text-muted">暂无审计结果</div>
          ) : (
            items.map(item => {
              const badge = statusBadge(item)
              return (
                <div key={item.id} className="border-b border-border/60 px-3 py-2 last:border-b-0">
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5">
                        {item.recommended && <CheckCircle2 className="h-3 w-3 shrink-0 text-accent" />}
                        <span className="truncate text-xs font-medium text-foreground">{item.label}</span>
                      </div>
                      <p className="mt-0.5 truncate text-[10px] text-muted" title={item.reason}>
                        {item.reason}
                      </p>
                    </div>
                    <div className="shrink-0 text-right">
                      <span className={`rounded px-1.5 py-px text-[10px] font-medium ${badge.cls}`}>
                        {badge.label}
                      </span>
                      <div className="mt-0.5 font-mono text-[10px] text-muted">
                        {rowsText(item)}
                      </div>
                    </div>
                  </div>
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}
