/**
 * The sample pages the challenge supplies, described for the UI.
 *
 * The filenames come from `vite.config.ts`, which reads the repository's `samples/` directory
 * at config time. Nothing here hard-codes a list: dropping a page into that directory adds it
 * to the picker, and deleting one removes it, which is how someone would expect it to work.
 * The descriptions below are keyed by filename and fall back to the filename itself, so an
 * unknown page still appears rather than being silently dropped.
 */

/** Injected by Vite's `define`; see the config for how it is derived. */
declare const __SAMPLE_FILES__: string[]
declare const __SAMPLE_URL_PREFIX__: string

export interface Sample {
  /** Filename as it exists in `samples/`, and as the ground truth indexes it. */
  file: string
  /** Short caption under the thumbnail. */
  label: string
  /** What the page is, used as the button's accessible name. */
  description: string
  thumb: string
  thumbWidth: number
  thumbHeight: number
}

/**
 * What makes each page interesting, which is the only reason to offer a choice of four.
 *
 * These are the four cases the detector was built against, and each defeated an earlier
 * version of it: the shaded rows broke global thresholding, the watermark broke a colour
 * assumption, and the zoomed crop broke a fixed pixel size range. Saying so turns the picker
 * from four buttons into a short tour of where the hard parts are.
 */
const DESCRIBED: Record<string, Omit<Sample, 'file' | 'thumb' | 'thumbWidth' | 'thumbHeight'>> = {
  'sample_1_urar_1004.png': {
    label: 'URAR 1004',
    description: 'a full Uniform Residential Appraisal Report page, 118 checkboxes',
  },
  'sample_2_crop_zoomed.jpeg': {
    label: 'Zoomed crop',
    description: 'a cropped region at a different scale, where a fixed pixel size range fails',
  },
  'sample_3_1004mc_shaded.png': {
    label: 'Shaded rows',
    description: 'a market-conditions form with shaded table rows, where global thresholding fails',
  },
  'sample_4_1004c_watermark.png': {
    label: 'Watermarked',
    description: 'a page under a red watermark, the hardest of the four',
  },
}

/** Thumbnail dimensions, so the strip reserves its space before the images decode. */
const THUMB_SIZE: Record<string, [number, number]> = {
  'sample_1_urar_1004.png': [260, 428],
  'sample_2_crop_zoomed.jpeg': [260, 139],
  'sample_3_1004mc_shaded.png': [260, 428],
  'sample_4_1004c_watermark.png': [260, 337],
}

function thumbUrl(file: string): string {
  // Thumbnails are JPEG regardless of the source format; see tools/make_sample_thumbs.py.
  return `${import.meta.env.BASE_URL}samples/${file.replace(/\.[^.]+$/, '')}.jpg`
}

export const SAMPLES: Sample[] = (typeof __SAMPLE_FILES__ === 'undefined' ? [] : __SAMPLE_FILES__)
  .map((file) => {
    const [width, height] = THUMB_SIZE[file] ?? [260, 340]
    return {
      file,
      thumb: thumbUrl(file),
      thumbWidth: width,
      thumbHeight: height,
      ...(DESCRIBED[file] ?? { label: file, description: file }),
    }
  })

/**
 * Fetch a full-size sample and hand it back as a File.
 *
 * A File rather than a URL so that a sample takes exactly the path an uploaded page does —
 * same validation, same object URL, same multipart request. Anything else would mean the
 * thing reviewers click is not quite the thing the app does.
 */
export async function loadSample(sample: Sample): Promise<File> {
  const prefix = typeof __SAMPLE_URL_PREFIX__ === 'undefined' ? '/sample-pages/' : __SAMPLE_URL_PREFIX__
  const response = await fetch(`${prefix}${sample.file}`)
  if (!response.ok) {
    throw new Error(`sample ${sample.file} responded ${response.status}`)
  }
  const blob = await response.blob()
  return new File([blob], sample.file, { type: blob.type || 'image/png' })
}
