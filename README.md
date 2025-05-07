# python-library
Here are some sample usages:
```python
split_by_semi_colon = raw_cookie_string.split(";")
for cookie_section in split_by_semi_colon:
    split_cookie = cookie_section.split("=")
    if len(split_cookie) > 2:
        return Err(NotImplementedError("Not ready for this"))
    cookie_name = split_cookie[0].strip()
    cookie_value = split_cookie[1].strip()
    cookies[cookie_name] = cookie_value
```
Refactored to:
```python
def split_cookie(cookie_section: str) -> Result[tuple[str, str]]:
    cookie_parts = cookie_section.split("=")
    if len(cookie_parts) > 2:
        return Err(NotImplementedError("Not ready for this"))
    return Ok(cookie_parts[0].strip(), cookie_parts[1].strip())

cookies = PyLinq(raw_cookie_string.split(";")) \
    .map(lambda c: split_cookie(c).unwrap()) \
    .to_dict(lambda c: c[0], lambda c: c[1])
```

