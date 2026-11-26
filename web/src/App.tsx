import { useEffect } from 'react'
import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom'

import Home from '@/pages/Home'
import Scan from '@/pages/Scan'

function ScrollToTop() {
  const { pathname } = useLocation()
  // The braces matter: an effect may only return a cleanup function, and scrollTo returns a promise in current Chrome.
  useEffect(() => { window.scrollTo(0, 0) }, [pathname])
  return null
}

export default function App() {
  return (
    <BrowserRouter>
      <ScrollToTop />
      <Routes>
        <Route path="/scan" element={<Scan />} />
        <Route path="*" element={<Home />} />
      </Routes>
    </BrowserRouter>
  )
}
