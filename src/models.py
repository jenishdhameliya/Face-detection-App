from django.db import models

# Create your models here.
app_name = 'src'
class Members(models.Model):
    name = models.CharField(max_length=250)
    age = models.IntegerField()
    def __str__(self):
        return  self.name

class Log(models.Model):
    entry = models.DateTimeField()
    member = models.ForeignKey(Members,on_delete=models.CASCADE)

    def __str__(self):
        return self.member.name