.PHONY: export tracking lint

export:
	python3 -m src.export

tracking_simple:
	python3 -m src.tracking.simple --video-path samples/highway.mp4

tracking_optimized:
	python3 -m src.tracking.optimized --video-path samples/highway.mp4

lint:
	pre-commit run --all-files
