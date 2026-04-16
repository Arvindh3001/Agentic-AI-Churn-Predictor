import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      // IBM Carbon Design System color tokens
      colors: {
        ibm: {
          "blue-10":   "#edf5ff",
          "blue-60":   "#0f62fe",
          "blue-70":   "#0043ce",
          "blue-80":   "#002d9c",
          "gray-100":  "#161616",
          "gray-90":   "#262626",
          "gray-80":   "#393939",
          "gray-70":   "#525252",
          "gray-60":   "#6f6f6f",
          "gray-50":   "#8d8d8d",
          "gray-30":   "#c6c6c6",
          "gray-20":   "#e0e0e0",
          "gray-10":   "#f4f4f4",
          "gray-10h":  "#e8e8e8",
          "red-60":    "#da1e28",
          "red-10":    "#fff1f1",
          "green-50":  "#24a148",
          "green-10":  "#defbe6",
          "yellow-30": "#f1c21b",
          "yellow-10": "#fcf4d6",
          "orange-40": "#ff832b",
          "orange-10": "#fff2e8",
        },
      },
      fontFamily: {
        sans: ["var(--font-ibm-plex-sans)", "Helvetica Neue", "Arial", "sans-serif"],
        mono: ["var(--font-ibm-plex-mono)", "Menlo", "Courier New", "monospace"],
      },
      borderRadius: {
        // Carbon is 0px everywhere. Only tags/pills use 24px (pill).
        DEFAULT: "0px",
        none:    "0px",
        sm:      "0px",
        md:      "0px",
        lg:      "0px",
        xl:      "0px",
        "2xl":   "0px",
        full:    "9999px",
        pill:    "24px",
      },
      boxShadow: {
        // Carbon is shadow-averse — only floating elements get shadows
        none:    "none",
        DEFAULT: "none",
        sm:      "none",
        md:      "0 2px 6px rgba(0,0,0,0.30)",
        lg:      "0 2px 6px rgba(0,0,0,0.30)",
        overlay: "0 2px 6px rgba(0,0,0,0.30)",
      },
    },
  },
  plugins: [],
};

export default config;
