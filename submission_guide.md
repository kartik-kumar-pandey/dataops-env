# OpenEnv Round 1 Submission Checklist

Here is the exact step-by-step process you need to follow to take your locally working environment, deploy it, and officially submit it.

## Phase 1: Local Pre-Check
Your code is fully compliant, but you can double check the environment logic one last time locally:
1. Open PowerShell and run this test to verify the baseline runs perfectly.
```powershell
$env:DOCKER_BUILDKIT="0"
docker run --rm dataops-env python inference.py
```
2. You should see the exact `[START]`, `[STEP]`, and `[END]` output with `score=1.00`.

## Phase 2: Configure Hugging Face Space
Since OpenEnv requires a live endpoint for the validators, you must deploy this directory to Hugging Face.
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. **Space name**: `dataops-env` (or whatever you like)
3. **Select Space SDK**: Choose **Docker** and select the Blank template. *(Extremely important)*
4. **Space Hardware**: The free tier (2 vCPU / 16GB) completely satisfies the contest constraints.
5. Click **Create Space**.

## Phase 3: Push Your Code
Push your local `hack/dataops-env` folder to the space. Assuming you have `git` installed, run these from your PowerShell prompt:
```bash
git init
git add .
git commit -m "Final OpenEnv DataOps submission with Dockerfile and requirements"
git remote add origin https://huggingface.co/spaces/<your-username>/dataops-env
git push -u origin main
```
*Note: Hugging Face will automatically detect the patched `Dockerfile`, build it, and launch your OpenEnv API.*

## Phase 4: Final Validation
Once Hugging Face says your Space is **"Running"**, grab your Space URL (e.g., `https://yourusername-dataops-env.hf.space`).
1. Run the official validator script using Git Bash or WSL in your local folder:
```bash
curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- https://yourusername-dataops-env.hf.space .
```
*(Make sure to update the github url and ping url to match the actual endpoints provided by the judges)*

2. The validator will ping your endpoint, test your `openenv.yaml` schema, and execute `python inference.py` to check the `[START]...[END]` outputs.

## Phase 5: Submit
If the `validate-submission.sh` script prints all green checkmarks and passes, you are 100% ready. Submit your Hugging Face Space URL and local Repository link to the hackathon portal!
