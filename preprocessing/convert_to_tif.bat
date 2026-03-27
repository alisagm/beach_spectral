@echo off
REM Convert all .sid and .ecw files to .tif using GDAL
REM Run this in OSGeo4W Shell

echo ============================================
echo Converting MrSID and ECW to GeoTIFF
echo ============================================

set COUNT=0

echo.
echo --- Converting .sid files ---
for /r "PAIS_shorelines\imagery" %%f in (*.sid) do (
    echo Converting: %%~nxf
    gdal_translate -of GTiff "%%f" "%%~dpnf.tif"
    if exist "%%~dpnf.tif" (
        echo   Success
        set /a COUNT+=1
    ) else (
        echo   FAILED
    )
)

echo.
echo --- Converting .ecw files ---
for /r "PAIS_shorelines\imagery" %%f in (*.ecw) do (
    echo Converting: %%~nxf
    gdal_translate -of GTiff "%%f" "%%~dpnf.tif"
    if exist "%%~dpnf.tif" (
        echo   Success
        set /a COUNT+=1
    ) else (
        echo   FAILED
    )
)

echo.
echo ============================================
echo Conversion complete. Files converted: %COUNT%
echo ============================================
pause