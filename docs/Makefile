# Minimal Sphinx makefile.
SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build, then serve the built site on the loopback interface only.
# Requires the docs environment on PATH (see docs/preview.md). Ctrl-C stops it.
PYTHON        ?= python
SERVE_PORT    ?= 8000

.PHONY: serve
serve: html
	@echo "Serving docs at http://127.0.0.1:$(SERVE_PORT)/  (Ctrl-C to stop)"
	@$(PYTHON) -m http.server $(SERVE_PORT) \
		--bind 127.0.0.1 --directory "$(BUILDDIR)/html"
