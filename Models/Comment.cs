using BurnAnalysisApp.Models;
public class Comment
{
    public int CommentID { get; set; }
    public int ForumPostID { get; set; } // Hangi gönderiye ait?
    public string DoctorName { get; set; } // Yorumu yazan doktor

    public string Content { get; set; } // Yorum içeriği
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

}