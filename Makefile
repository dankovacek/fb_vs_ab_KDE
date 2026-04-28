.PHONY: readme

# Render README.md from README.src.md using Pandoc + citeproc.
# Inline citations (e.g. [@silverman2018density]) are formatted as author-year
# and a References section is appended automatically.
readme:
	pandoc README.src.md \
		--citeproc \
		--bibliography=references.bib \
		-t gfm \
		-o README.md
