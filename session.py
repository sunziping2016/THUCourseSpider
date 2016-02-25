import urllib.parse, urllib.request
import http.cookiejar as cookiejar
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class Session(object):
    def __init__(self):
        self.cookie = cookiejar.LWPCookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cookie))
    def headers(self):
        return {'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6',
                'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Encoding':'deflate,sdch',
                'Accept-Language':'en-US,en;q=0.8,zh-CN;q=0.6,zh;q=0.4'}
    def get(self, url, data = None):
        if data:
            url += '?' + urllib.parse.urlencode(data)
        req = urllib.request.Request(url = url, headers = self.headers())
        resp = self.opener.open(req)
        return resp
    def post(self, url, data):
        req = urllib.request.Request(url = url, data = urllib.parse.urlencode(data).encode('utf-8'), headers = self.headers())
        req.add_header('Content-Type', 'application/x-www-form-urlencoded');
        resp = self.opener.open(req)
        return resp
    def clear(self):
        self.cookie.clear_session_cookies()
