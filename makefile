# make dev name=keras_model
dev:
	bq module install --dev --name $(name)

undev:
	bq module uninstall --dev --dev --name $(name)

publish:
	bq module publish --name $(name)
	# git push --tags


all_dev:
	@for dir in $(shell ls -d ./src/*/); do \
		name=$$(basename $$dir); \
		$(MAKE) dev name=$$name; \
	done


all_undev:
	@for dir in $(shell ls -d ./src/*/); do \
		name=$$(basename $$dir); \
		$(MAKE) undev name=$$name; \
	done


all_publish:
	@for dir in $(shell ls -d ./src/*/); do \
		name=$$(basename $$dir); \
		$(MAKE) publish name=$$name; \
	done
