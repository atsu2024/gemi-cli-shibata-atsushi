# Repository Guidelines

## Project Structure & Module Organization

This repository combines a Node.js API, browser assets, and scientific computing programs in C, Go, and Python. The main server entry point is `server.js`, with database setup in `database.js` and browser files under `public/` (`index.html`, `app.js`, `style.css`). Core compiled sources live in `src/c/` and `src/go/`; Python utilities are in `src/python/`. Root-level `.c` and `.go` files are also built by the Makefile. Build outputs belong in `bin/`. Documentation and research inputs are stored in `docs/`, JSON, CSV, PDF, and image files.

## Build, Test, and Development Commands

- `npm install`: install Node dependencies from `package-lock.json`.
- `npm run dev`: start `server.js` with `nodemon` for local development.
- `npm start`: run the API with Node.
- `npm run build`: run `mingw32-make -f Makefile all` to compile C and Go programs into `bin/`.
- `npm test`: run `node test_api.js`; start the server first on `http://localhost:3000`.
- `npm run clean`: remove compiled `.exe` files and `bin/libdnn.o`.
- `npm run lint`: placeholder only; no lint tool is currently configured.

## Coding Style & Naming Conventions

Use CommonJS style for Node files (`require`, `module.exports`) and keep JavaScript indentation at two spaces. Use `gofmt` for Go files and conventional `snake_case` names for C source files, matching existing files such as `scientific_precision_ld.c`. Keep generated binaries out of source folders and place compiled artifacts under `bin/`. Prefer descriptive simulation names that match the executable names exposed by the API.

## Testing Guidelines

The current test entry point is `test_api.js`, an Axios-based API smoke test. It expects the server to be running locally and compiled simulations to exist in `bin/`, so run `npm run build` and `npm start` before `npm test`. Add new endpoint checks to `test_api.js` or create similarly named `test_*.js` files for focused coverage.

## Commit & Pull Request Guidelines

Git history currently contains only `Initial commit: High-Precision Scientific Computing & Deep Learning project`, so use clear imperative commit messages such as `Add batch simulation endpoint` or `Fix Lorenz build flags`. Pull requests should describe the change, list build/test commands run, note new data or generated artifacts, and link related issues. Include screenshots when changing `public/` UI behavior.

## Security & Configuration Tips

Keep secrets in `.env` and do not commit API keys, local databases, or generated binaries unless they are required fixtures. Review changes to `research.db`, PDFs, and large data files carefully before committing.
