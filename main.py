import urllib.parse, urllib.request
import getpass
import ssl
import captcha
import re

ssl._create_default_https_context = ssl._create_unverified_context

try:
    import cookielib
except Exception as e:
    import http.cookiejar as cookielib

def SettingUp():
    Cookie = cookielib.LWPCookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(Cookie))
    urllib.request.install_opener(opener)

def GetHeaders():
    header = {'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6',
                'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Encoding':'deflate,sdch',
                'Accept-Language':'en-US,en;q=0.8,zh-CN;q=0.6,zh;q=0.4'}
    return header

def Post(url, data):
    req = urllib.request.Request(url = url, data = urllib.parse.urlencode(data).encode('utf-8'), headers = GetHeaders())
    req.add_header('Content-Type', 'application/x-www-form-urlencoded');
    try:
        response = urllib.request.urlopen(req)
    except urllib.request.URLError as e:
        print(e)
        exit()
    return response
def Get(url, data = None):
    if data:
        url += '?' + urllib.parse.urlencode(data)
    req = urllib.request.Request(url = url, headers = GetHeaders())
    try:
        response = urllib.request.urlopen(req)
    except urllib.request.URLError as e:
        print(e)
        exit()
    return response

SettingUp()

captchacache = captcha.CaptchaCache("Captcha.txt")
captchacache.load()

url = 'http://zhjwxk.cic.tsinghua.edu.cn'

if Get(url + '/xklogin.do').status != 200:
    print('ERROR.')
    exit()

image = Get(url + '/login-jcaptcah.jpg?captchaflag=login1').read()
imagefile = open('captcha.jpeg', 'wb')
imagefile.write(image)
imagefile.close()

#username = input('Username: ')
#password = getpass.getpass()
username = 'szp15'
password = '3000years!'

res = captchacache.search(image)
if res:
    code = res
else:
    code = input('Captcha: ')
    code = code.upper()

data = {'j_username': username,
        'j_password': password,
        'captchaflag': 'login1',
        '_login_image_': code}

res = Post('https://zhjwxk.cic.tsinghua.edu.cn/j_acegi_formlogin_xsxk.do', data)
if res.status != 200:
    print('ERROR.')
    exit()

content = res.read().decode('gb2312')
error = re.findall(r'<div align="center">(.*?)</div>', content, re.S)

if len(error)!=0:
    print('ERROR: ' + error[0])
    exit()
else:
    print('Login Success!')
    captchacache.add(image, code)
    captchacache.save()


