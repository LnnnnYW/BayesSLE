# real_data/prep_adni_baseline.R
prep_adni_baseline <- function(
  data,
  target="MMSCORE",
  id_cols = c("RID","PTID"),
  feature_exclude = c("PHASE","SITEID","ID"),
  task = c("auto","classification","regression"),
  na_col_drop_thresh = 0.5,
  cat_max_levels = 5
){
  suppressPackageStartupMessages({ library(dplyr); library(tidyr) })

  task <- match.arg(task)
  stopifnot(target %in% names(data))
  if (task == "auto")
    task <- if (is.numeric(data[[target]])) "regression" else "classification"

  df <- data

  # drop high-NA columns among candidate features
  drop_cols <- unique(c(id_cols, feature_exclude, target))
  cand_cols <- setdiff(names(df), drop_cols)
  if (length(cand_cols)) {
    na_rate <- sapply(df[cand_cols], function(v) mean(is.na(v)))
    drop_by_na <- names(na_rate)[na_rate > na_col_drop_thresh]
    if (length(drop_by_na)) df <- df %>% dplyr::select(-dplyr::any_of(drop_by_na))
  }

  # target must be non-NA
  df <- df %>% dplyr::filter(!is.na(.data[[target]]))

  # 1) baseline de-dup per subject
  cand_dates <- intersect(
    c("EXAMDATE","EXAMDATE_bl","USERDATE","SCANDATE","VISITDATE","EXAM_DT"),
    names(df)
  )
  if ("VISCODE" %in% names(df)) {
    df$.is_bl <- tolower(as.character(df$VISCODE)) %in% c("bl","sc","init","baseline")
    df <- df %>%
      dplyr::group_by(.data[[id_cols[1]]]) %>%
      dplyr::arrange(dplyr::desc(.is_bl), .by_group = TRUE) %>%
      dplyr::slice(1) %>%
      dplyr::ungroup() %>%
      dplyr::select(-.is_bl)
  } else if (length(cand_dates)) {
    dcol <- cand_dates[1]
    df[[dcol]] <- suppressWarnings(as.Date(df[[dcol]]))
    df <- df %>%
      dplyr::group_by(.data[[id_cols[1]]]) %>%
      dplyr::arrange(.data[[dcol]], .by_group = TRUE) %>%
      dplyr::slice(1) %>%
      dplyr::ungroup()
  } else {
    df <- df %>% dplyr::group_by(.data[[id_cols[1]]]) %>% dplyr::slice(1) %>% dplyr::ungroup()
  }

  # 2) drop rows with any NA among feature columns only
  feat_for_na <- setdiff(names(df), unique(c(id_cols, feature_exclude, target)))
  if (length(feat_for_na))
    df <- df %>% dplyr::filter(complete.cases(dplyr::across(dplyr::all_of(feat_for_na))))

  # 3) assemble X / y
  X_raw <- df %>% dplyr::select(-dplyr::any_of(unique(c(id_cols, feature_exclude, target))))
  y     <- df[[target]]

  # type split: numeric with low cardinality -> categorical; others -> continuous
  is_char <- sapply(X_raw, is.character)
  is_fact <- sapply(X_raw, is.factor)
  is_num  <- sapply(X_raw, is.numeric)

  num_cols_all <- names(X_raw)[is_num]
  num_unique <- if (length(num_cols_all))
    sapply(X_raw[num_cols_all], function(v) length(unique(v))) else integer(0)
  num_as_cat <- names(num_unique)[num_unique > 0 & num_unique <= cat_max_levels]
  num_cont   <- setdiff(num_cols_all, num_as_cat)

  cat_cols  <- union(names(X_raw)[is_char | is_fact], num_as_cat)
  cont_cols <- num_cont

  # categorical: one-hot; single-level factor -> all-ones column
  X_cat <- NULL
  if (length(cat_cols)) {
    X_cat_df <- X_raw[cat_cols]
    for (cc in cat_cols) {
      if (!is.factor(X_cat_df[[cc]])) X_cat_df[[cc]] <- factor(X_cat_df[[cc]])
      X_cat_df[[cc]] <- droplevels(X_cat_df[[cc]])
    }
    lev_counts <- vapply(X_cat_df, nlevels, integer(1))
    multi_fac_cols  <- names(lev_counts)[lev_counts >= 2L]
    single_fac_cols <- names(lev_counts)[lev_counts == 1L]

    X_cat_multi <- if (length(multi_fac_cols)) {
      as.data.frame(stats::model.matrix(~ . - 1, data = X_cat_df[multi_fac_cols]),
                    check.names = FALSE)
    } else NULL

    X_cat_single <- if (length(single_fac_cols)) {
      tmp <- lapply(single_fac_cols, function(cc){
        lev <- levels(X_cat_df[[cc]])[1]
        setNames(data.frame(rep(1, nrow(X_cat_df)), check.names = FALSE),
                 paste0(cc, lev))
      })
      as.data.frame(do.call(cbind, tmp), check.names = FALSE)
    } else NULL

    X_cat <- dplyr::bind_cols(X_cat_multi, X_cat_single)
    if (!is.null(X_cat) && !ncol(X_cat)) X_cat <- NULL
  }

  # continuous: z-score; keep sd==0 columns unchanged
  X_cont <- NULL
  if (length(cont_cols)) {
    X_cont <- X_raw[cont_cols]
    for (cc in cont_cols) {
      sdv <- stats::sd(X_cont[[cc]])
      if (is.finite(sdv) && sdv > 0) {
        X_cont[[cc]] <- (X_cont[[cc]] - mean(X_cont[[cc]])) / sdv
      }
    }
  }

  X_proc <- dplyr::bind_cols(X_cont, X_cat) %>% as.data.frame()

  # checks
  stopifnot(nrow(X_proc) > 1, ncol(X_proc) > 0)
  if (any(!is.finite(as.matrix(X_proc)))) stop("X contains NA/NaN/Inf after preprocessing.")
  if (task == "regression") {
    if (!is.numeric(y)) stop("For regression, y must be numeric.")
    if (!all(is.finite(y))) stop("y contains NA/NaN/Inf.")
    if (isTRUE(stats::sd(y) == 0)) stop("y is constant; not learnable.")
  } else {
    if (!is.factor(y)) y <- factor(y)
    if (nlevels(y) < 2) stop("Classification y must have at least 2 classes.")
  }

  list(
    X = as.matrix(X_proc),
    y = if (task == "classification") factor(y) else as.numeric(y),
    task = task,
    feature_names = colnames(X_proc),
    kept_n = nrow(df)
  )
}