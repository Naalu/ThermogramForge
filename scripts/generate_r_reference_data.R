# Script to generate reference data from R's smooth.spline
# for validation testing

# Function to generate and save test data with different patterns
generate_test_data <- function(pattern, n = 100, output_prefix) {
    set.seed(42) # For reproducibility

    # Generate x values and synthetic data based on pattern
    if (pattern == "sine") {
        x <- seq(0, 10, length.out = n)
        y <- sin(x) + rnorm(n, 0, 0.1)
    } else if (pattern == "exponential") {
        x <- seq(0, 5, length.out = n)
        y <- exp(x / 2) + rnorm(n, 0, 0.5)
    } else if (pattern == "peaks") {
        # Thermogram-like pattern with multiple peaks
        x <- seq(45, 90, length.out = n)
        y <- dnorm(x, mean = 55, sd = 2) + dnorm(x, mean = 70, sd = 3) +
            dnorm(x, mean = 82, sd = 2) + rnorm(n, 0, 0.02)
    } else if (pattern == "flat") {
        # Nearly constant data with small variations
        x <- seq(0, 10, length.out = n)
        y <- rep(5, n) + rnorm(n, 0, 0.1)
    } else if (pattern == "noisy") {
        # Very noisy data
        x <- seq(0, 10, length.out = n)
        y <- sin(x) + rnorm(n, 0, 0.5)
    }

    # Sort values (required for spline fitting)
    idx <- order(x)
    x <- x[idx]
    y <- y[idx]

    # Fit with smooth.spline
    fit <- smooth.spline(x, y, cv = TRUE)

    # Get fitted values
    fitted_values <- predict(fit, x)$y

    # Save input data
    write.csv(data.frame(x = x, y = y),
        file = paste0(output_prefix, "_", pattern, "_input.csv"),
        row.names = FALSE
    )

    # Save fitted values
    write.csv(data.frame(x = x, fitted = fitted_values),
        file = paste0(output_prefix, "_", pattern, "_fitted.csv"),
        row.names = FALSE
    )

    # Save spline parameters
    params <- data.frame(
        spar = fit$spar,
        df = fit$df,
        lambda = fit$lambda
    )
    write.csv(params,
        file = paste0(output_prefix, "_", pattern, "_params.csv"),
        row.names = FALSE
    )

    cat("Generated data for pattern:", pattern, "\n")
    cat("  - Smoothing parameter (spar):", fit$spar, "\n")
    cat("  - Effective degrees of freedom:", fit$df, "\n")
    cat("  - Lambda:", fit$lambda, "\n\n")

    return(list(
        x = x, y = y, fitted = fitted_values,
        spar = fit$spar, df = fit$df, lambda = fit$lambda
    ))
}

# Output directory
output_dir <- "tests/data/r_reference/"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Generate different test cases
patterns <- c("sine", "exponential", "peaks", "flat", "noisy")
for (pattern in patterns) {
    # Generate data with 100 points
    result <- generate_test_data(
        pattern,
        n = 100, paste0(output_dir, "standard")
    )

    # Also generate a small dataset version for testing with few points
    if (pattern == "sine" || pattern == "peaks") {
        small_result <- generate_test_data(
            pattern,
            n = 20, paste0(output_dir, "small")
        )
    }
}

# Generate real thermogram-like data with different parameter settings
x_thermo <- seq(45, 90, length.out = 200)
set.seed(123)
y_thermo <- dnorm(x_thermo, mean = 55, sd = 2.5) +
    dnorm(x_thermo, mean = 67, sd = 1.8) +
    dnorm(x_thermo, mean = 78, sd = 2.2) +
    rnorm(length(x_thermo), 0, 0.01)

# Save raw thermogram data
write.csv(data.frame(x = x_thermo, y = y_thermo),
    file = paste0(output_dir, "thermogram_raw.csv"),
    row.names = FALSE
)

# Fit with different cv settings
fit_cv_true <- smooth.spline(x_thermo, y_thermo, cv = TRUE)
fit_cv_false <- smooth.spline(x_thermo, y_thermo, cv = FALSE)

# Save fitted values
write.csv(data.frame(
    x = x_thermo,
    y = y_thermo,
    fitted_cv_true = predict(fit_cv_true, x_thermo)$y,
    fitted_cv_false = predict(fit_cv_false, x_thermo)$y
), file = paste0(output_dir, "thermogram_fits.csv"), row.names = FALSE)

# Save parameters
params <- data.frame(
    cv = c("TRUE", "FALSE"),
    spar = c(fit_cv_true$spar, fit_cv_false$spar),
    df = c(fit_cv_true$df, fit_cv_false$df),
    lambda = c(fit_cv_true$lambda, fit_cv_false$lambda)
)
write.csv(params,
    file = paste0(output_dir, "thermogram_params.csv"),
    row.names = FALSE
)

cat("Generated thermogram reference data\n")
cat(
    "CV=TRUE parameters:",
    fit_cv_true$spar,
    fit_cv_true$df,
    fit_cv_true$lambda,
    "\n"
)
cat(
    "CV=FALSE parameters:",
    fit_cv_false$spar,
    fit_cv_false$df,
    fit_cv_false$lambda,
    "\n"
)

cat("Reference data generation complete\n")
