import type { Metadata } from 'next'
import { Noto_Sans_Telugu, Inter } from 'next/font/google'
import './globals.css'

const notoSansTelugu = Noto_Sans_Telugu({ 
  subsets: ['telugu', 'latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-telugu'
})

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter'
})

export const metadata: Metadata = {
  title: 'Telugu Sarcasm Dataset Collection',
  description: 'Help us build a comprehensive Telugu sarcasm detection database',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${notoSansTelugu.variable} ${inter.variable}`}>
        {children}
      </body>
    </html>
  )
}
