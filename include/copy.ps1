# Create or overwrite the output file
$output_file = "all_code.txt"

# Add a header with timestamp
"// C++ Files Concatenated on $(Get-Date)" | Set-Content $output_file
"// ----------------------------------------" | Add-Content $output_file

# Function to process files of a specific extension from a directory
function Add-Files-From-Directory {
    param (
        [string]$path,
        [string]$filter
    )
    if (Test-Path $path) {
        Get-ChildItem -Path $path -Filter $filter -Recurse | ForEach-Object {
            "`n`n// File: $($_.FullName)" | Add-Content $output_file
            "// ----------------------------------------" | Add-Content $output_file
            Get-Content $_.FullName | Add-Content $output_file
        }
    }
}

# Process .hpp files from dl directory
Add-Files-From-Directory -path "dl" -filter "*.hpp"

# Process .cpp files from examples directory (up one level)
Add-Files-From-Directory -path "../examples" -filter "*.cpp"

# Process .cpp files from tests directory (up one level)
Add-Files-From-Directory -path "../tests" -filter "*.cpp"

Write-Host "All .hpp and .cpp files have been concatenated into $output_file"