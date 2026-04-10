# https://mc-stan.org/cmdstanr/articles/articles-online-only/opencl.html

library(mirai)

# prerequisites
# install cmdstan
cmdstanr::install_cmdstan()

# set OpenCL flag for compilation
cpp_options = list(
  "LDFLAGS+= -L  /usr/local/cuda-12.4/lib64/ -lOpenCL"
)

# rebuild cmdstan with OpenCL enabled
cmdstanr::cmdstan_make_local(cpp_options = cpp_options)
cmdstanr::rebuild_cmdstan()

# Get STAN model
write(
  RCurl::getURI(
    "https://raw.githubusercontent.com/stan-dev/cmdstanr/master/vignettes/articles-online-only/opencl-files/bernoulli_logit_glm.stan"
  ),
  "model.stan"
)

# Generate some fake data
n <- 250000
k <- 20
X <- matrix(rnorm(n * k), ncol = k)
y <- rbinom(n, size = 1, prob = plogis(3 * X[, 1] - 2 * X[, 2] + 1))
mdata <- list(k = k, n = n, y = y, X = X)

# Compile model on the GPU using OpenCL
mod_cl <- cmdstanr::cmdstan_model(
  "model.stan",
  cpp_options = list(stan_opencl = TRUE)
)

# Set up 2 daemons for parallel benchmarking
daemons(2)

# Run fit_cl 2 times in parallel for GPU benchmarking
results <- mirai_map(
  1:2,
  function(run_id, mod, data) {
    start_time <- Sys.time()

    fit <- mod$sample(
      data = data,
      chains = 4,
      parallel_chains = 4,
      opencl_ids = c(0, 0),
      refresh = 0
    )

    end_time <- Sys.time()
    elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))

    list(
      run_id = run_id,
      elapsed_seconds = elapsed,
      fit = fit
    )
  },
  .args = list(mod = mod_cl, data = mdata)
)[.progress]

# Clean up daemons
daemons(0)

# Print benchmark results
cat("\nGPU Benchmark Results:\n")
cat("=====================\n")
for (result in results) {
  cat(sprintf("Run %d: %.2f seconds\n", result$run_id, result$elapsed_seconds))
}
cat(sprintf("\nMean: %.2f seconds\n", mean(sapply(results, `[[`, "elapsed_seconds"))))
cat(sprintf("SD: %.2f seconds\n", sd(sapply(results, `[[`, "elapsed_seconds"))))

