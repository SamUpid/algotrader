/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'trading-green': '#00d4aa',
        'trading-red': '#ff6b6b',
        'trading-bg': '#0a0e17',
        'trading-card': '#141c2b',
        'trading-border': '#1e2d45',
      }
    },
  },
  plugins: [],
}
