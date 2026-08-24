import { useState } from 'react'
import { SAMPLES, loadSample, type Sample } from '../lib/samples'

interface SamplePickerProps {
  /** Receives the fetched page as a File, exactly as the file input would. */
  onPick: (file: File) => void
  /** Filename of the currently loaded page, so the matching thumbnail can be marked. */
  selected?: string
  disabled?: boolean
}

/**
 * The four supplied pages, one click away.
 *
 * The reason this exists is that judging a detector means looking at its output on a real
 * document, and until now that took finding the repository on disk and dragging a file in.
 * A reviewer opening the hosted demo had nothing to try it on at all.
 *
 * A thumbnail is a small committed JPEG; the full page — up to 2550x4200 — is fetched only
 * when one is chosen. Shipping four full pages to render them at 120px would put megabytes in
 * front of the first paint of a demo that scales to zero.
 *
 * Picking loads the page but does NOT detect, which is the same rule the file input follows.
 * The rule is worth keeping even now that detection is free: a page is several seconds of CPU,
 * and a click that starts work the user did not ask for is a click they cannot take back.
 */
export function SamplePicker({ onPick, selected, disabled }: SamplePickerProps) {
  const [loading, setLoading] = useState<string | null>(null)
  const [failed, setFailed] = useState<string>('')

  async function choose(sample: Sample) {
    if (disabled || loading) return
    setLoading(sample.file)
    setFailed('')
    try {
      onPick(await loadSample(sample))
    } catch {
      // Named rather than generic: the one way this fails in practice is the page being
      // opened from a build where the sample directory was not copied in, and "sample_3
      // could not be loaded" points at that far better than "something went wrong".
      setFailed(`${sample.label} could not be loaded.`)
    } finally {
      setLoading(null)
    }
  }

  if (SAMPLES.length === 0) return null

  return (
    <div className="samples">
      <span className="label" id="samples-label">
        Try a sample page
      </span>

      <div className="samples__grid" role="group" aria-labelledby="samples-label">
        {SAMPLES.map((sample) => {
          const isLoading = loading === sample.file
          return (
            <button
              key={sample.file}
              type="button"
              className={`samples__item${selected === sample.file ? ' samples__item--active' : ''}`}
              onClick={() => void choose(sample)}
              disabled={disabled || loading !== null}
              // The visible caption is an abbreviation ("Watermarked"); the accessible name
              // says what the page actually is, since a screen reader user cannot see it.
              aria-label={`Load ${sample.description}`}
              aria-busy={isLoading}
            >
              <img
                src={sample.thumb}
                alt=""
                loading="lazy"
                width={sample.thumbWidth}
                height={sample.thumbHeight}
              />
              <span className="samples__caption">{isLoading ? 'Loading…' : sample.label}</span>
            </button>
          )
        })}
      </div>

      {failed && <span className="notice notice--error">{failed}</span>}
    </div>
  )
}
