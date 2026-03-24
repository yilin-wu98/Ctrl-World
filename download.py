from huggingface_hub import snapshot_download
snapshot_download(
  repo_id = 'yilin-wu/world-model-droid-eval',
  repo_type='dataset',
  local_dir = '/scratch/jarnav/SAILOR/DROID/world-model-droid-eval-new',
  resume_download=True,
  max_workers=4,
  )
