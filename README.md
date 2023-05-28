## Fast usage

You can tinker with the DALL-E playground using a Github-hosted frontend. Follow these steps:

1. Run the DALL-E backend using Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
2. Copy the URL from the last executed cell. Look for the line having `your url is: https://XXXX.trycloudflare.com`
3. **Wait for the backend to fully load**, this should take ~2min and you should see `--> DALL-E Server is up and running!`
5. Enter the URL in the app settings.

**General note**: while it is possible to run the DALL-E Mini backend on the free tier of Google Colab,
generating more than 1-2 images would take more than 1min, which will result in a frontend timeout. Consider upgrading to Colab Pro or run the backend notebook on your stronger ML machine (e.g. AWS EC2).

## Using DALL-E Mega
DALL-E Mega is substantially more capable than DALL-E Mini and therefore generates higher fidelity images. If you have the computing power--either through a Google Colab Pro+ subcription or by having a strong local machine, select the DALL-E Mega model in the colab notebook or run the backend with a `Mega` or `Mega_full` parameter, e.g. `python dalle-playground/backend/app.py --port 8000 --model_version mega`

## Acknowledgements

[Boris Dayma's](https://github.com/borisdayma) DALL-E Mini repository. 
