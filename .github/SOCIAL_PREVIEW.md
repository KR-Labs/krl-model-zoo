# GitHub Social Preview Image

## What is a Social Preview Image?

The social preview image is what appears when your repository is shared on social media (Twitter, LinkedIn, Slack, etc.) or in GitHub search results.

## Current Image

The repository should use: `assets/images/KRLabs_WebLogo.png`

## Optimal Specifications

- **Dimensions:** 1280 x 640 pixels (2:1 ratio)
- **Format:** PNG or JPG
- **File Size:** Under 1 MB
- **Content:** Should include:
  - KR-Labs logo
  - Repository name: "KRL Model Zoo"
  - Tagline: "Production-Ready Econometric & ML Models"
  - Key features or model types

## How to Upload

1. Go to: https://github.com/KR-Labs/krl-model-zoo/settings
2. Scroll to "Social preview" section
3. Click "Edit"
4. Upload the image
5. Click "Save"

## Design Recommendations

### Text to Include
```
KR-Labs Model Zoo
Production-Ready Models for:
• Econometrics (ARIMA, GARCH, VAR)
• Time Series Forecasting
• Machine Learning (RF, XGBoost)
• Causal Inference
• Regional Analysis

100+ Models | Apache 2.0 | Python
```

### Color Scheme
- Use KR-Labs brand colors
- High contrast for readability
- Professional, clean design

### Logo Placement
- Top left or center
- Ensure it's clearly visible at thumbnail size

## Alternative: Use GitHub Generated Image

If no custom image is uploaded, GitHub will auto-generate one showing:
- Repository name
- Description
- Language breakdown
- Stars/Forks count

## Testing the Preview

After uploading, test how it looks by:
1. Sharing the repo URL on Twitter/LinkedIn
2. Using a social media preview tool like:
   - https://www.opengraph.xyz/
   - https://cards-dev.twitter.com/validator

## Image Assets Location

Store social preview assets in:
```
assets/social/
├── github-social-preview.png (1280x640)
├── github-social-preview-light.png
└── github-social-preview-dark.png
```
