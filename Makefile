.PHONY: test milestone1 audit milestone2a milestone2b milestone2c milestone3a milestone3b milestone3c milestone3d export-web api

test:
	pytest

milestone1:
	quant-research run-milestone1

audit:
	quant-research audit-milestone1

milestone2a:
	quant-research run-milestone2a

milestone2b:
	quant-research run-milestone2b

milestone2c:
	quant-research run-milestone2c

milestone3a:
	quant-research run-milestone3a

milestone3b:
	quant-research run-milestone3b

milestone3c:
	quant-research run-milestone3c

milestone3d:
	quant-research run-milestone3d

export-web:
	quant-research export-web

api:
	quant-research serve-api
