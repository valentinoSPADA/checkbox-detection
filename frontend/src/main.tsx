/** React entry point. */
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'

const container = document.getElementById('root')
if (!container) {
  // Fail loudly rather than rendering nothing: a silent blank page is the hardest kind of
  // frontend failure to diagnose from a screenshot.
  throw new Error('#root element is missing from index.html')
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
