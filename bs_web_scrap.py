from bs4 import BeautifulSoup
import re


def scrap_cid(body: str):
    signature_body = re.findall(
        r'<div id=\".*?Signature\">',
        body)

    a = body.split(signature_body[0])[1]
    print(a)
    soup = BeautifulSoup(a, features="html.parser")

    image_tags = soup.findAll('img')
    print(f'image_tags: {image_tags}')
    signature_images = [image.attrs.get('alt') for image in image_tags]

    print(signature_images)


if __name__ == "__main__":
    conversation_body = """
    <html><head>
        <meta content="text/html; charset=utf-8" http-equiv="Content-Type"/></head><body><font face="Arial">Hello, Thank you for messaging here!</font><hr style="display:inline-block; width:98%" tabindex="-1"/><div dir="ltr" id="divRplyFwdMsg"><font color="#000000" face="Calibri, sans-serif" style="font-size:11pt"><b>From:</b> Prasanna L &lt;prasanna.l@transcenddigital.com&gt;<br/><b>Sent:</b> Thursday, January 5, 2023 4:48:51 AM<br/><b>To:</b> Prasana L &lt;prasanna20212022@outlook.com&gt;<br/><b>Subject:</b> Re: Test Mail for broken signature</font> <div> </div></div><div><div dir="ltr">Thank you for messaging me, Much needed to hear from you</div><br/><div class="gmail_quote"><div class="gmail_attr" dir="ltr">On Thu, Jan 5, 2023 at 1:18 PM Prasana L &lt;<a href="mailto:prasanna20212022@outlook.com">prasanna20212022@outlook.com</a>&gt; wrote:<br/></div><blockquote class="gmail_quote" style="margin:0px 0px 0px 0.8ex; border-left:1px solid rgb(204,204,204); padding-left:1ex"><div class="msg-9151634139056348139"><div dir="ltr"><div><span style="font-family:Calibri,Arial,Helvetica,sans-serif; font-size:12pt; color:rgb(0,0,0); background-color:rgb(255,255,255)">Hello 001</span></div><div><div style="font-family:Calibri,Arial,Helvetica,sans-serif; font-size:12pt; color:rgb(0,0,0)"><br/></div><div id="m_-9151634139056348139Signature"><div><div style="font-family:Calibri,Arial,Helvetica,sans-serif; font-size:12pt; color:rgb(0,0,0); background-color:rgb(255,255,255)"><span style='font-family:"Lucida Console",Monaco,monospace'>Regards,</span></div><div style="font-family:Calibri,Arial,Helvetica,sans-serif; font-size:12pt; color:rgb(0,0,0); background-color:rgb(255,255,255)"><span style='font-family:"Lucida Console",Monaco,monospace'>Prasanna L</span></div><div style="font-family:Calibri,Arial,Helvetica,sans-serif; font-size:12pt; color:rgb(0,0,0); background-color:rgb(255,255,255)"><span style='font-family:"Lucida Console",Monaco,monospace'><br/></span></div><div style="font-family:Calibri,Arial,Helvetica,sans-serif; font-size:12pt; color:rgb(0,0,0); background-color:rgb(255,255,255)"><span style='font-family:"Lucida Console",Monaco,monospace'><a href="https://google.com" target="_blank" title="https://google.com"><img height="152" id="m_-9151634139056348139imageSelected0" src="cid:18580e739ee90bdddcf1" style="max-width:initial; width:103px; height:152px" width="103"/></a><img height="147" id="m_-9151634139056348139imageSelected3" src="cid:18580e739ee1ae6ea222" style="max-width:initial; width:98.6768px; height:147px" width="98"/><img height="121" id="m_-9151634139056348139imageSelected1" src="cid:18580e739ee86403b9b3" style="max-width:initial; width:73.8737px; height:121px" width="73"/><br/></span></div></div></div></div></div></div></blockquote></div></div></body></html>
    """
    scrap_cid(conversation_body)
