from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import webview


def launch_html_viewer( html: str,
                        title: str = "HTML Viewer",
                        width: int = 1200,
                        height: int = 800,
                        *,
                        use_temp_file: bool = True,
                        debug: bool = False, ) -> None:
    """
    Launch a native desktop window that renders arbitrary HTML/JS.

    Arguments:
    ----------
    html:
        Full HTML document as a string.
    title:
        Window title.
    width, height:
        Initial window size.
    use_temp_file:
        If True, writes HTML to a temporary .html file and loads it via file://.
        This is often more robust for JS/WebGL libraries than passing html=...
    debug:
        If True, enables pywebview debug mode.
    """
    if use_temp_file:
        tmp_dir = Path(tempfile.mkdtemp(prefix="html_viewer_"))
        html_path = tmp_dir / "index.html"
        html_path.write_text(html, encoding="utf-8")

        window = webview.create_window(
            title=title,
            url=html_path.as_uri(),
            width=width,
            height=height,
        )
    else:
        window = webview.create_window(
            title=title,
            html=html,
            width=width,
            height=height,
        )

    webview.start(debug=debug)

def make_page( molecule_html_blocks: list[str], phi : list, psi : list, energies : list ) -> str:
    molecule_cards = "\n".join(
        f"""
        <div class="card">
          <h2>Conformer {i} (phi,psi) = ({phi[i]:.1f}, {psi[i]:.1f}), E = {energies[i]:.2f} eV</h2>
          {block}
        </div>
        """
        for i, block in enumerate(molecule_html_blocks) )

    return f"""
        <!doctype html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: system-ui, sans-serif;
                margin: 24px;
                background: #f7f7f7;
            }}

            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(540px, 1fr));
                gap: 20px;
            }}

            .card {{
                background: white;
                border-radius: 14px;
                padding: 16px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.12);
            }}
        </style>
        </head>
        <body>
        <h1>Conformers</h1>
        <div class="grid">
            {molecule_cards}
        </div>
        </body>
        </html>
        """