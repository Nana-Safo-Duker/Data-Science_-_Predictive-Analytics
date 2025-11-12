# Helper function for path resolution in R scripts
# This file contains common functions for finding project root and data/output paths

# Function to find project root by looking for data directory
find_project_root <- function() {
  # Start from current working directory
  current_dir <- getwd()
  max_levels <- 10
  project_marker <- file.path("Consumer Purchase Prediction", "Consumer Purchase Prediction", "data", "Advertisement.csv")
  
  for (i in 1:max_levels) {
    # Check for the data file in nested structure
    if (file.exists(file.path(current_dir, project_marker))) {
      return(current_dir)
    }
    # Check for data file in current directory structure
    if (file.exists(file.path(current_dir, "data", "Advertisement.csv"))) {
      return(current_dir)
    }
    # Check if we're in the Consumer Purchase Prediction directory
    if (basename(current_dir) == "Consumer Purchase Prediction") {
      if (file.exists(file.path(current_dir, "Consumer Purchase Prediction", "data", "Advertisement.csv"))) {
        return(current_dir)
      }
      if (file.exists(file.path(current_dir, "data", "Advertisement.csv"))) {
        return(current_dir)
      }
    }
    # Move up one level
    parent_dir <- dirname(current_dir)
    if (parent_dir == current_dir) break  # Reached root
    current_dir <- parent_dir
  }
  return(NULL)
}

# Function to find data file
find_data_file <- function() {
  project_root <- find_project_root()
  if (is.null(project_root)) {
    project_root <- getwd()
  }
  
  # Try multiple possible paths
  data_paths <- c(
    file.path(project_root, "Consumer Purchase Prediction", "Consumer Purchase Prediction", "data", "Advertisement.csv"),
    file.path(project_root, "data", "Advertisement.csv"),
    file.path(project_root, "Advertisement.csv")
  )
  
  for (path in data_paths) {
    if (file.exists(path)) {
      return(path)
    }
  }
  
  return(NULL)
}

# Function to get output directory
get_output_dir <- function() {
  project_root <- find_project_root()
  if (is.null(project_root)) {
    project_root <- getwd()
  }
  
  # Try multiple possible output paths
  output_paths <- c(
    file.path(project_root, "Consumer Purchase Prediction", "Consumer Purchase Prediction", "output"),
    file.path(project_root, "output")
  )
  
  # Check if output directory exists
  for (path in output_paths) {
    if (dir.exists(path)) {
      return(path)
    }
  }
  
  # Create the first possible output directory
  output_dir <- output_paths[1]
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  return(output_dir)
}

# Initialize paths (call this at the start of each script)
initialize_paths <- function() {
  project_root <- find_project_root()
  if (!is.null(project_root)) {
    setwd(project_root)
    cat("Working directory set to:", getwd(), "\n")
  } else {
    cat("Warning: Could not find project root. Using current directory:", getwd(), "\n")
  }
  
  data_path <- find_data_file()
  if (is.null(data_path)) {
    stop(paste("Cannot find Advertisement.csv. Current working directory:", getwd()))
  }
  
  output_dir <- get_output_dir()
  
  return(list(data_path = data_path, output_dir = output_dir))
}

