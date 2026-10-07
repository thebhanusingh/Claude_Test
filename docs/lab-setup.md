# Lab setup: Gaussian Splats across Claude Code, Blender, and Unreal Engine

Goal: work on the same Gaussian Splat, interchangeably, across three tools — the
video→splat pipeline in this repo, Blender (for editing/rendering), and Unreal
Engine (for real-time/relightable viewing) — on a university lab machine where
you don't control the firewall and don't have admin rights, without needing
either.

The organizing principle: **everything lives inside your own user profile /
project folder.** Nothing here needs to write to `Program Files`, `/usr`,
the registry, or any other admin-only location, *provided the big GUI apps
(Unreal Engine itself, and ideally Blender) are already installed on the lab
image by IT.* That one item is the only thing you may not be able to do
yourself — see below.

## Checklist

- [ ] **Unreal Engine already installed on the lab machine** — check with IT/lab
      staff rather than assuming. If it's there, you never touch the installer.
      If it isn't, ask them to image it or grant a one-time admin install —
      this is the one piece that's plausibly out of your hands.
- [ ] **Blender** — use the portable ZIP build from blender.org (not the
      installer/Microsoft Store version). Unzip into your own user folder and
      run it directly; nothing is installed system-wide, no admin needed.
- [ ] **KIRI Engine's "3DGS Render" Blender addon** — installs through Blender's
      own Preferences UI, writes only to your Blender config folder (inside
      your user profile). No admin.
- [ ] **An Unreal Gaussian Splat plugin** — copied into *your own project's*
      `Plugins/` folder (inside your project directory, which lives wherever
      you have write access — Documents, a personal drive, a USB/portable
      SSD). No admin, as long as you pick a plugin that doesn't require
      recompiling the engine (see below).
- [ ] **conda/miniforge + this repo's pipeline** — already covered by
      `scripts/setup_env.sh`; installs entirely into your home directory.
- [ ] **Claude Code CLI locally** (optional, if you want it running directly
      on the lab machine instead of only this web session) — see below.
- [ ] **Outbound HTTPS reachable** to: `github.com`, `pypi.org`,
      `anaconda.org`/conda-forge mirrors, `download.pytorch.org`, and (if you
      install Unreal via the Epic Games Launcher yourself) Epic's CDN, plus
      `api.anthropic.com` if running Claude Code locally. University firewalls
      commonly allow plain outbound HTTPS (443) but may block specific hosts
      or require a proxy — if a clone/install hangs or gets refused, that's
      the first thing to check with IT, not a sign the tool is broken.

You do **not** need admin rights for any of this as long as Unreal Engine
itself is already on the machine — everything else is either a portable app,
a user-scoped install, or a file copied into a folder you already own.

## 1. Blender: KIRI Engine's "3DGS Render" addon

Free, open source: [Kiri-Innovation/3dgs-render-blender-addon](https://github.com/Kiri-Innovation/3dgs-render-blender-addon).
It imports `.ply` Gaussian Splats and renders them natively in Eevee, composited
with normal Blender scene elements.

1. Download the portable Blender ZIP (blender.org → Download → look for the
   `.zip`, not the installer) and unzip it into your own folder.
2. Download the addon's release ZIP from the GitHub repo above (don't clone —
   Blender addons install from a zip via the UI).
3. In Blender: `Edit > Preferences > Add-ons > Install...`, pick the zip,
   enable it. It now appears as a "3DGS Render" tab in the sidebar (`N`
   panel).
4. `Import PLY` to load a splat exported by `scripts/make_splat.sh`
   (`exports/<scene>/*.ply`). Use "Import as Points" first for fast
   positioning, then switch to full splat rendering.

## 2. Unreal Engine: a Gaussian Splat plugin

Recommended: **[xverse-engine/XScene-UEPlugin](https://github.com/xverse-engine/XScene-UEPlugin)**
— Apache 2.0 (commercial use OK), UE 5.0+, built on UE's built-in Niagara VFX
system. Because it's Niagara-based rather than requiring you to compile new
C++ against the engine, it should **not** need Visual Studio / the Windows
SDK / a C++ toolchain — the single biggest admin-rights landmine on a locked
lab machine. Verify this on the actual plugin release you grab (check its
release notes for a precompiled/binary drop vs. "build from source").

Other options if this one doesn't fit your UE version: **NanoGS** (free,
Nanite-style LOD for large splat scenes — ships a plugin you drop into
`Plugins/` and import through) and **XV3DGS-UEPlugin**. Steer away from
anything that requires an Epic Marketplace/Fab purchase or account if you
want to minimize network/account dependencies on a restricted network.

Install:
1. `git clone https://github.com/xverse-engine/XScene-UEPlugin` somewhere in
   your own writable space.
2. Copy the plugin folder into `YourProject/Plugins/`.
3. In the editor: `Edit > Plugins`, enable **Niagara** (built into the
   engine, just a toggle) and the XVERSE3DGS plugin, restart the editor when
   prompted.
4. Keep your UE project **Blueprint-only** (don't add a C++ class) unless a
   plugin specifically forces a source build — this avoids needing Visual
   Studio entirely. If a plugin only ships as source and won't work
   precompiled, that's the point to ask IT whether the C++ toolchain is
   already on the machine, rather than trying to install it yourself.

## 3. Splat format compatibility

`scripts/make_splat.sh` exports a standard Inria/3DGS-format `.ply`
(position, opacity, scale, rotation, spherical-harmonics color per splat) via
nerfstudio's `ns-export gaussian-splat`. Both the Blender addon and the UE
plugins above are built against that same de facto format, so the exported
`.ply` should import into either directly — but "should" is doing some work
there across independently-developed tools; do a quick test import before
relying on it for anything time-sensitive.

## 4. Claude Code across all three

This repo/project folder is the natural hub: Claude Code already drives the
splat pipeline (`scripts/*.sh`) here, and can drive the other two the same
way once you're at the lab machine:

- **Blender**, headless: `blender --background --python your_script.py` —
  Claude Code can write/run a script that imports a `.ply`, sets up
  materials/render settings, and renders or re-exports, without you touching
  the Blender UI.
- **Unreal Engine**, headless/scripted: enable UE's built-in **Python Editor
  Script Plugin** (`Edit > Plugins`, no admin, it ships with the engine) and
  drive it via `UnrealEditor-Cmd.exe YourProject.uproject -run=pythonscript
  -script="..."` or the in-editor Python console — asset import, level
  setup, etc. can be scripted the same way as the splat pipeline.

### Running Claude Code CLI directly on the lab machine (optional)

If you want Claude Code itself running locally there (rather than only this
cloud session), install Node.js in **user space** to avoid needing admin —
e.g. via `nvm` (Node Version Manager), which installs entirely inside your
home directory, then `npm install -g @anthropic-ai/claude-code` with npm's
global prefix pointed at a folder you own. Check Claude Code's own docs for
the current recommended install method for your platform before doing this,
since install steps change over time.

## Notes on drivers/GPU

Not a concern here per your setup — lab GPUs are already on current drivers.
`scripts/setup_env.sh` still prints the detected driver/GPU and warns only if
something is unusually old (Blackwell-class cards need R570+); on a
already-current machine this check is a no-op.
