# Deployment Guide - AI Fitness Coach

## 🚀 Deploying to Streamlit Cloud

### 1. Required Files (Already Setup)

Make sure these files are in **the same directory as app.py**:
- ✅ `app.py` - Main application
- ✅ `requirements.txt` - Python dependencies (simplified)
- ✅ `packages.txt` - System dependencies (minimal)
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `src/` directory - All source code

**Important:** If your repo has subdirectories (e.g., `ai-workout/weightlifting-ai/`), put these files in the **same folder as app.py**, not the repo root!

### 2. Push to GitHub

```bash
cd weightlifting-ai  # Navigate to the folder with app.py
git add .
git commit -m "Simplify dependencies for cloud deployment"
git push
```

### 3. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository
4. **IMPORTANT:** Set main file path correctly:
   - If repo structure is: `repo-name/weightlifting-ai/app.py`
   - Then set path to: `weightlifting-ai/app.py`
5. Click "Deploy"

### 4. CRITICAL - File Locations

Streamlit Cloud looks for `requirements.txt` and `packages.txt` in the **same directory** as your main Python file!

**Example repo structure:**
```
your-repo/
├── weightlifting-ai/          ← Your app folder
│   ├── app.py                ← Main file
│   ├── requirements.txt      ← MUST be here (same level as app.py)
│   ├── packages.txt          ← MUST be here (same level as app.py)
│   ├── .streamlit/
│   │   └── config.toml
│   └── src/
│       ├── analysis/
│       └── exercises/
```

### 4. Important Notes

**CAMERA LIMITATIONS:**
- ❌ Live camera does NOT work on Streamlit Cloud (server has no camera)
- ✅ Video upload DOES work perfectly
- 💡 App automatically detects cloud and disables camera mode

**FOR USERS:**
- Record videos on phone/laptop
- Upload to the cloud app
- Get instant analysis and feedback

**FOR DEVELOPMENT:**
- Run locally for live camera: `streamlit run app.py`
- Test cloud features: upload mode works the same locally

## 🐛 Troubleshooting

### OpenCV Import Error (Most Common!)

If you see this error:
```
ImportError: ... cv2 ... bootstrap() ... cannot open shared object file
```

**Fix:**

1. **Check file locations:**
   ```bash
   cd weightlifting-ai  # Folder with app.py
   ls -la requirements.txt packages.txt
   ```
   These files MUST be in the same folder as app.py!

2. **Verify file contents:**

   **requirements.txt** should contain:
   ```
   streamlit
   opencv-python-headless
   mediapipe
   numpy<2.0.0
   protobuf<4.0.0
   ```

   **packages.txt** should contain:
   ```
   libgl1
   libglib2.0-0
   ```

3. **Delete old files:**
   - Remove any `requirement.txt` (singular)
   - Remove any old `requirements.txt` with pinned versions
   - Start fresh with simple versions above

4. **In Streamlit Cloud dashboard:**
   - Click "Manage app" (bottom right)
   - Click "Reboot app"
   - Watch the logs during deployment

5. **Check the build logs:**
   - Look for: "Installing packages from packages.txt"
   - Look for: "Installing Python packages from requirements.txt"
   - If you don't see these, files are in wrong location!

### Memory Issues

If app crashes due to memory:
1. Use smaller video files (< 50MB)
2. Lower resolution videos work better
3. Streamlit Cloud has 1GB RAM limit

### Reboot App

After fixing issues:
1. Go to app settings
2. Click "Reboot app"
3. Or redeploy from GitHub

## 📊 Performance Optimization

**For Best Cloud Performance:**
- Accept videos up to 30 seconds
- Recommend 720p or lower resolution
- MP4 format is most compatible

## ✅ Deployment Checklist

- [ ] Deleted old `requirement.txt` file
- [ ] Only `requirements.txt` exists
- [ ] Uses `opencv-python-headless`
- [ ] Pushed latest changes to GitHub
- [ ] Rebooted Streamlit Cloud app
- [ ] Tested video upload functionality
- [ ] Verified all 11 exercises work

## 🆘 Still Having Issues?

Check Streamlit Cloud logs:
1. Click "Manage app" (bottom right)
2. View logs for detailed errors
3. Look for package installation errors
