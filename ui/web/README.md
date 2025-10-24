Graph AML Investigator UI
=========================

Quick start
-----------

- Prerequisites: Node 18+ and npm.
- The backend should be running on http://localhost:8000 with CORS enabled (this repo already enables it for Vite).

Commands
--------

1) Install deps

   npm i

2) Start dev server (Vite on http://localhost:5173)

   npm run dev

3) Build for production (outputs to `ui/web/dist`)

   npm run build

4) Preview production build locally

   npm run preview

Auth (optional)
---------------

If the backend uses bearer auth, set a token in localStorage:

   localStorage.setItem('API_TOKEN', 'your-token')

Or use the token input box in the top-right of the UI header.

Config
------

- Set `VITE_API_BASE` to point the UI to a reverse-proxied API if needed. Defaults to `window.location.origin`.
