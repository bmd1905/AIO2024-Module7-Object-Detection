.PHONY: export tracking lint

export:
	python3 -m src.export

tracking:
	python3 -m src.tracking --video-path samples/vietnam-1.mp4

lint:
	pre-commit run --all-files
