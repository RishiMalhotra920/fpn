# Makefile to create a flat zip of fpn directory excluding dot files and __pycache__


# The directory to zip
SOURCE_DIR := fpn

# Temporary directory for flattening
TEMP_DIR := temp_flat

# Default target
all: flat

# Target to create the flat zip file
flat:
	@echo "Creating flat DIR from $(SOURCE_DIR) directory..."
	@rm -rf $(TEMP_DIR)
	@mkdir $(TEMP_DIR)
	@find $(SOURCE_DIR) -type f \
		! -path "*/\.*" \
		! -path "*/__pycache__/*" \
		! -name "*.pyc" \
		-exec cp {} $(TEMP_DIR) \;
	@echo "$(ZIP_NAME) created successfully."

# Clean target to remove the zip file
clean:
	@echo "Removing $(ZIP_NAME)..."
	@rm -rf $(TEMP_DIR)
	@echo "$(TEMP_DIR) removed."

# Help target to display available commands
help:
	@echo "Available commands:"
	@echo "  make           - Create the flat zip file (same as 'make zip_fpn_flat')"
	@echo "  make zip_fpn_flat - Create the flat zip file"
	@echo "  make clean      - Remove the zip file"
	@echo "  make help       - Display this help message"

.PHONY: all zip_fpn_flat clean help