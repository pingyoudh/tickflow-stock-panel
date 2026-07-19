import { beforeEach, describe, expect, it } from 'vitest'
import { storage } from '@/lib/storage'
import { getCardVisibility } from './PageSettingsModal'

describe('data page ETF card visibility', () => {
  beforeEach(() => localStorage.clear())

  it('shows ETF by default', () => {
    expect(getCardVisibility({ 'kline.daily.batch': {} }).etf).toBe(true)
  })

  it('migrates the old hidden ETF default once', () => {
    storage.dataCardVisible.set({ etf: false, minute: false })

    const visible = getCardVisibility({ 'kline.daily.batch': {} })

    expect(visible.etf).toBe(true)
    expect(visible.minute).toBe(false)
    expect(storage.dataCardVisibilityVersion.get(0)).toBe(3)
  })

  it('respects a user choice made after migration', () => {
    storage.dataCardVisibilityVersion.set(2)
    storage.dataCardVisible.set({ etf: false })

    expect(getCardVisibility({ 'kline.daily.batch': {} }).etf).toBe(false)
  })

  it('shows newly registered news and depth cards after migration', () => {
    storage.dataCardVisibilityVersion.set(2)
    storage.dataCardVisible.set({ minute: false })

    const visible = getCardVisibility({})

    expect(visible.finance_news).toBe(true)
    expect(visible.depth5).toBe(true)
    expect(visible.minute).toBe(false)
  })
})
