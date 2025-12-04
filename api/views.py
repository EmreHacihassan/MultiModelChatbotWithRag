from django.http import JsonResponse

def health(_req):
  return JsonResponse({"ok": True, "name": "MyChatbot Backend", "ts": "live"})
