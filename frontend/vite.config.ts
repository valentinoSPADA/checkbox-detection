import { createReadStream, existsSync, cpSync, readdirSync } from 'node:fs'
import { basename, join, resolve } from 'node:path'
import { defineConfig, type Plugin } from 'vitest/config'
import react from '@vitejs/plugin-react'

/** The sample pages the challenge supplies, at the repository root. */
const SAMPLES_DIR = resolve(__dirname, '..', 'samples')

/** URL prefix the UI fetches a full-size sample from. */
const SAMPLE_URL_PREFIX = '/sample-pages/'

/**
 * Serves and ships the sample pages from the repository root, without copying them into the
 * frontend.
 *
 * They already live in `samples/`: the README's curl uses them, `eval/evaluate.py` scores
 * against them, and the ground truth is indexed by their filenames. A second copy under
 * `public/` would be 2.8 MB that can silently drift from the one every other part of the
 * project measures against — the kind of divergence nobody notices until a number stops
 * reproducing.
 *
 * Two hooks, because dev and build resolve files differently: middleware answers requests
 * from disk while the dev server runs, and `writeBundle` copies the directory into `dist/`
 * for the production image, where the Go binary embeds and serves it.
 */
function samplePages(): Plugin {
  return {
    name: 'sample-pages',

    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith(SAMPLE_URL_PREFIX)) return next()

        // basename() and nothing else: the URL segment must not be able to name a path
        // outside this directory, and the dev server has the whole working tree beneath it.
        const requested = decodeURIComponent(req.url.slice(SAMPLE_URL_PREFIX.length).split('?')[0])
        const file = join(SAMPLES_DIR, basename(requested))

        if (!existsSync(file)) {
          res.statusCode = 404
          res.end('no such sample page')
          return
        }
        createReadStream(file).pipe(res)
      })
    },

    writeBundle(options) {
      if (!existsSync(SAMPLES_DIR)) return
      cpSync(SAMPLES_DIR, join(options.dir ?? 'dist', 'sample-pages'), { recursive: true })
    },
  }
}

/**
 * Sample filenames, read from disk at config time.
 *
 * Baked into the bundle rather than hard-coded in a component, so the picker cannot offer a
 * page that is not there — adding or removing a sample is a matter of dropping a file in the
 * directory, which is how someone would expect it to work.
 */
const sampleFiles = existsSync(SAMPLES_DIR)
  ? readdirSync(SAMPLES_DIR)
      .filter((name) => /\.(png|jpe?g)$/i.test(name))
      .sort()
  : []

// The dev server binds 0.0.0.0 so the container-published port is reachable from the host;
// binding localhost inside a container makes the app appear dead from outside it.
export default defineConfig({
  plugins: [react(), samplePages()],

  define: {
    __SAMPLE_FILES__: JSON.stringify(sampleFiles),
    __SAMPLE_URL_PREFIX__: JSON.stringify(SAMPLE_URL_PREFIX),
  },

  server: { host: '0.0.0.0', port: 5173 },
  preview: { host: '0.0.0.0', port: 5173 },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test-setup.ts'],
    coverage: { provider: 'v8', reporter: ['text', 'lcov'], include: ['src/**/*.{ts,tsx}'] },
  },
})
