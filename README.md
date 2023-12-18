# sphaera

sphaera - a math toolkit for spherical data processing

![wind velocity](wind-velocity.png)

How to release
---------------

```bash
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*

git tag va.b.c master
git push origin va.b.c
```
