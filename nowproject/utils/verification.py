import xarray as xr
from xverif import xverif

def verification_routine(ds_forecast, ds_obs, skills_dir, print_metrics=True):
    ds_det_cont_skill = xverif.deterministic(
            pred=ds_forecast.chunk({"time": -1}),
            obs=ds_obs.sel(time=ds_forecast.time).chunk({"time": -1}),
            forecast_type="continuous",
            aggregating_dim="time",
        )
        # - Save sptial skills
    ds_det_cont_skill.to_netcdf((skills_dir / "deterministic_continuous_spatial_skill.nc"))

    thresholds = [0.1, 1, 5, 10, 15]
    ds_det_cat_skills = {}
    ds_det_spatial_skills = {}
    for thr in thresholds:
        print("Threshold:", thr)
        ds_det_cat_skill = xverif.deterministic(
            pred=ds_forecast.chunk({"time": -1}),
            obs=ds_obs.sel(time=ds_forecast.time).chunk({"time": -1}),
            forecast_type="categorical",
            aggregating_dim="time",
            thr=thr
        )
        # - Save sptial skills
        ds_det_cat_skill.to_netcdf((skills_dir / f"deterministic_categorical_spatial_skill_thr{thr}.nc"))
        ds_det_cat_skills[thr] = ds_det_cat_skill

        spatial_scales = [5]
        ds_det_spatial_skills[thr] = {}
        for spatial_scale in spatial_scales:
            print("Spatial scale:", spatial_scale)
            ds_det_spatial_skill = xverif.deterministic(
                pred=ds_forecast.chunk({"x": -1, "y": -1}),
                obs=ds_obs.sel(time=ds_forecast.time).chunk({"x": -1, "y": -1}),
                forecast_type="spatial",
                aggregating_dim=["x", "y"],
                thr=thr,
                win_size=spatial_scale
            )
            ds_det_spatial_skill.to_netcdf((skills_dir/ f"deterministic_spatial_skill_thr{thr}_scale{spatial_scale}.nc"))
            ds_det_spatial_skills[thr][spatial_scale] = ds_det_spatial_skill

    ### - Compute averaged skill scores
    ds_cont_averaged_skill = ds_det_cont_skill.mean(dim=["y", "x"])
    ds_cont_averaged_skill.to_netcdf(skills_dir/ "deterministic_continuous_global_skill.nc")

    ds_cat_averaged_skills = {}
    ds_spatial_averaged_skills = {}
    for thr in ds_det_cat_skills:
        ds_cat_averaged_skills[thr] = ds_det_cat_skills[thr].mean(dim=["y", "x"])
        ds_cat_averaged_skills[thr].to_netcdf(skills_dir / f"deterministic_categorical_global_skill_thr{thr}_mean.nc")
        
        ds_spatial_averaged_skills[thr] = {}
        for spatial_scale in ds_det_spatial_skills[thr]:
            ds_spatial_averaged_skills[thr][spatial_scale] = ds_det_spatial_skills[thr][spatial_scale].mean(dim=["time"])
            ds_spatial_averaged_skills[thr][spatial_scale].to_netcdf(skills_dir / f"deterministic_spatial_global_skill_thr{thr}_scale{spatial_scale}.nc")

    if print_metrics:
        print("RMSE:")
        print(ds_cont_averaged_skill["feature"].sel(skill="RMSE").values)

        for thr in ds_cat_averaged_skills:
            print(f"\nCategorical and Spatial metrics for threshold {thr}\n")
            print(f"F1@{thr}:")
            print(ds_cat_averaged_skills[thr]["feature"].sel(skill="F1").values)
            print(f"ACC@{thr}:")
            print(ds_cat_averaged_skills[thr]["feature"].sel(skill="ACC").values)
            print(f"CSI@{thr}:")
            print(ds_cat_averaged_skills[thr]["feature"].sel(skill="CSI").values)
            for spatial_scale in ds_spatial_averaged_skills[thr]:
                print(f"FSS@{thr} threshold and @{spatial_scale} spatial scale:")
                print(ds_spatial_averaged_skills[thr][spatial_scale]["feature"].sel(skill="FSS").values)
    
    return ds_det_cont_skill, ds_det_cat_skills, ds_det_spatial_skills, \
            ds_cont_averaged_skill, ds_cat_averaged_skills, ds_spatial_averaged_skills