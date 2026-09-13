from dataclasses import dataclass
TEMPLATES = {}
@dataclass
class Template:
    format_user: str
    format_assistant: str
    def format(self,user_content,assistant_content):
        prompt=self.format_user.replace("{content}",user_content)
        response=self.format_assistant.replace("{content}",assistant_content)
        full_text=prompt+response
        return prompt,full_text
    def format_prompt(self,user_content):
        prompt=self.format_user.replace("{content}",user_content)
        return prompt
def register_template(name,format_user,format_assistant):
    if name in TEMPLATES:
        raise ValueError(f"Template {name} already exists.")
    template=Template(format_user=format_user,format_assistant=format_assistant)
    TEMPLATES[name]=template
    
register_template(
    name="qwen",
    format_user="{content}",
    format_assistant="{content}<|endoftext|>\n",
)